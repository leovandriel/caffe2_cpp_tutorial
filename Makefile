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
	@rm -rf include/res

test: all
	@find bin -regex "bin/[a-z]*" -type f -exec ./{} \;

# for filename in bin/*; do;./${filename}|diff test/$(echo ${filename} | cut -d/ -f2).log -;done

output: all
	@find bin -regex "bin/[a-z]*" -type f -exec bash -c './{} > test/$(echo {} | cut -d/ -f2).log' \;
