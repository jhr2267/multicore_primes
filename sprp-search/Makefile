all: parallel

seq:
	g++ -std=c++0x -o program primesieve/*.cpp sprp_search.cpp RandMT.cpp mypair.cpp

parallel: sprp_search_cuda.o
	nvcc -Xcompiler="-std=c++0x" -o program primesieve/*.cpp sprp_search_cuda.cpp RandMT.cpp sprp_search_cuda.o mypair.o

sprp_search_cuda.o:
	nvcc -c -arch=compute_35 -code=sm_35 sprp_search_cuda.cu mypair.cpp

run: 
	./program

clean:
	rm -rf *.o program
