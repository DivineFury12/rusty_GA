# ---------- Build stage ----------
FROM rust:latest AS builder

WORKDIR /app

COPY Cargo.toml ./

# Cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

# Copy real source
COPY src ./src
COPY *.csv ./

# Final build
RUN cargo build --release


# ---------- Runtime stage ----------
FROM debian:bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/rusty_GA ./rusty_GA
COPY *.csv ./

CMD ["./rusty_GA"]
