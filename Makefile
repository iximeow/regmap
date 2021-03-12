build: example.s
	cargo build --release --workspace
	as examples/example.s -o target/example.o
	gcc target/example.o target/release/libregmap.a -o asm_example

joy: build
	./asm_example
