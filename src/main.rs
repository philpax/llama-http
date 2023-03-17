use std::sync::{Arc, Barrier};

use axum::{extract::State, routing::post, Json, Router};
use clap::Parser;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Where to load the model path from
    #[arg(long, short = 'm')]
    pub model_path: String,

    /// Sets the number of threads to use
    #[arg(long, short = 't', default_value_t = std::thread::available_parallelism().unwrap().get())]
    pub num_threads: usize,

    /// Sets the size of the context (in tokens). Allows feeding longer prompts.
    /// Note that this affects memory. TODO: Unsure how large the limit is.
    #[arg(long, default_value_t = 512)]
    pub num_ctx_tokens: usize,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let barrier = Arc::new(Barrier::new(2));

    let (request_tx, request_rx) = flume::unbounded::<ThreadGenerateRequest>();
    let _model_thread = std::thread::spawn({
        let barrier = barrier.clone();
        move || {
            let (model, vocab) = llama_rs::Model::load(
                &args.model_path,
                args.num_ctx_tokens.try_into().unwrap(),
                |progress| {
                    use llama_rs::LoadProgress;
                    match progress {
                        LoadProgress::HyperParamsLoaded(hparams) => {
                            println!("Loaded HyperParams {hparams:#?}")
                        }
                        LoadProgress::BadToken { index } => {
                            println!("Warning: Bad token in vocab at index {index}")
                        }
                        LoadProgress::ContextSize { bytes } => println!(
                            "ggml ctx size = {:.2} MB\n",
                            bytes as f64 / (1024.0 * 1024.0)
                        ),
                        LoadProgress::MemorySize { bytes, n_mem } => println!(
                            "Memory size: {} MB {}",
                            bytes as f32 / 1024.0 / 1024.0,
                            n_mem
                        ),
                        LoadProgress::PartLoading {
                            file,
                            current_part,
                            total_parts,
                        } => println!(
                            "Loading model part {}/{} from '{}'\n",
                            current_part,
                            total_parts,
                            file.to_string_lossy(),
                        ),
                        LoadProgress::PartTensorLoaded {
                            current_tensor,
                            tensor_count,
                            ..
                        } => {
                            if current_tensor % 8 == 0 {
                                println!("Loaded tensor {current_tensor}/{tensor_count}");
                            }
                        }
                        LoadProgress::PartLoaded {
                            file,
                            byte_size,
                            tensor_count,
                        } => {
                            println!("Loading of '{}' complete", file.to_string_lossy());
                            println!(
                                "Model size = {:.2} MB / num tensors = {}",
                                byte_size as f64 / 1024.0 / 1024.0,
                                tensor_count
                            );
                        }
                    }
                },
            )
            .unwrap();

            barrier.wait();

            let mut rng = rand::thread_rng();
            loop {
                if let Ok(ThreadGenerateRequest(request, token_tx)) = request_rx.try_recv() {
                    model.inference_with_prompt(
                        &vocab,
                        &llama_rs::InferenceParameters {
                            n_threads: args.num_threads.try_into().unwrap(),
                            n_predict: request.num_predict,
                            n_batch: request.batch_size,
                            top_k: request.top_k.try_into().unwrap(),
                            top_p: request.top_p,
                            repeat_last_n: request.repeat_last_n,
                            repeat_penalty: request.repeat_penalty,
                            temp: request.temp,
                        },
                        &request.prompt,
                        &mut rng,
                        {
                            let token_tx = token_tx.clone();
                            move |t| {
                                token_tx
                                    .send(match t {
                                        llama_rs::OutputToken::Token(t) => {
                                            Token::Token(t.to_string())
                                        }
                                        llama_rs::OutputToken::EndOfText => Token::EndOfText,
                                    })
                                    .unwrap();
                            }
                        },
                    );
                };

                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    });

    barrier.wait();

    let app = Router::new()
        .route("/generate", post(generate))
        .with_state(request_tx);

    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}

enum Token {
    Token(String),
    EndOfText,
}

struct ThreadGenerateRequest(GenerateRequest, flume::Sender<Token>);

#[derive(Deserialize)]
#[serde(default)]
struct GenerateRequest {
    /// The prompt to feed the generator
    pub prompt: String,

    /// Sets how many tokens to predict
    pub num_predict: usize,

    /// How many tokens from the prompt at a time to feed the network. Does not
    /// affect generation.
    pub batch_size: usize,

    /// Size of the 'last N' buffer that is used for the `repeat_penalty`
    /// option. In tokens.
    pub repeat_last_n: usize,

    /// The penalty for repeating tokens. Higher values make the generation less
    /// likely to get into a loop, but may harm results when repetitive outputs
    /// are desired.
    pub repeat_penalty: f32,

    /// Temperature
    pub temp: f32,

    /// Top-K: The top K words by score are kept during sampling.
    pub top_k: usize,

    /// Top-p: The cummulative probability after which no more words are kept
    /// for sampling.
    pub top_p: f32,
}
impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            prompt: Default::default(),
            num_predict: 128,
            batch_size: 8,
            repeat_last_n: 64,
            repeat_penalty: 1.30,
            temp: 0.80,
            top_k: 40,
            top_p: 0.95,
        }
    }
}

#[derive(Serialize)]
struct GenerateResponse {
    text: String,
}

async fn generate(
    State(request_tx): State<flume::Sender<ThreadGenerateRequest>>,
    Json(request): Json<GenerateRequest>,
) -> Json<GenerateResponse> {
    let (token_tx, token_rx) = flume::unbounded();

    request_tx
        .send(ThreadGenerateRequest(request, token_tx))
        .unwrap();

    let mut text = String::new();
    let mut stream = token_rx.into_stream();

    while let Some(token) = stream.next().await {
        match token {
            Token::Token(t) => {
                text += t.as_str();
            }
            Token::EndOfText => {}
        }
    }

    Json(GenerateResponse { text })
}
