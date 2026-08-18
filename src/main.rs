#[tokio::main(flavor = "current_thread")]
async fn main() {
    match a_agent::cli::run().await {
        Ok(0) => {}
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("error: {error:#}");
            std::process::exit(1);
        }
    }
}
