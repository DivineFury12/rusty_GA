# ---------- Build stage ----------
FROM rust:1.71 as builder

WORKDIR /app

# Copy manifest first for dependency caching
COPY Cargo.toml Cargo.lock ./

# Create dummy main.rs to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

RUN cargo build --release
RUN rm -rf src

# Copy real source code (main.rs only, parallel ignored via .dockerignore)
COPY src ./src
COPY *.csv ./

# Build release binary
RUN cargo build --release


# ---------- Runtime stage ----------
FROM debian:bookworm-slim

WORKDIR /app

# Install minimal runtime deps
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=builder /app/target/release/* ./

# Copy CSV files
COPY *.csv ./

# Run app
CMD ["./main"]