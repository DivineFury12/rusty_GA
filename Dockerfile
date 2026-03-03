FROM rust:latest

WORKDIR /app

# Copy dependency manifests first (for better caching)
COPY Cargo.toml ./

# Copy source code and data files
COPY src ./src
COPY *.csv ./

# Build the release binary (caches dependencies)
RUN cargo build --release

# Run the program using cargo run --release
CMD ["cargo", "run", "--release"]
