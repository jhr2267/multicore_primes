all: parallel

seq:
	g++ -std=c++11 miller-rabin.cpp

parallel:
	nvcc -o program miller-rabin.cu

run:
	./program

clean:
	rm -rf *.o *.obj program
