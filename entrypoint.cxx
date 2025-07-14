#include "cmut.hxx"
#include <iostream>

int main() noexcept {

	cmut<std::uint64_t> mut(4096 * 4096);

	auto s = sizeof(cmut<std::uint64_t>);

	mut /= 4096;

	mut += 8;

	mut -= 4;
	
	std::cout << mut.get() << std::endl;

	return 0;
}