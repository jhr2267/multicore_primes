all: parallel

seq: sprp_search.o
	g++ -std=c++0x -o program *.o
	
sprp_search.o:
	g++ -std=c++0x -c primesieve/*.cpp sprp_search.cpp RandMT.cpp mypair.cpp

parallel: sprp_search_cuda.o
	nvcc -g -G -Xcompiler="-std=c++0x" -o program primesieve/*.cpp sprp_search_cuda.cpp RandMT.cpp sprp_search_cuda.o mypair.o

sprp_search_cuda.o:
	nvcc -g -c -arch=compute_35 -code=sm_35 sprp_search_cuda.cu mypair.cpp

run:
	./program

clean:
	rm -rf *.o program
