# Proxy to CMake

all:
	@test -d build || (mkdir -p build && cd build && cmake .. && cd .. && ./script/download_resource.sh); cd build && make

clean:
	@rm -rf build

purge:
	@rm -rf build
	@rm -rf bin
	@rm -rf res
	@rm -rf tmp
	@rm include/imagenet_classes.h