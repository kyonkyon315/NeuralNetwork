#include "common.h"
//https://www.youtube.com/@ProgrammerCpp 様からの引用

std::istream& operator>>(std::istream& istm, char&& r) {
	char ch;
	if (isgraph(r) ? istm >> ch : istm.read(&ch, 1)) {
		//文字が読めた
		if (ch != r) {
			//異常（別の文字だった）
			istm.setstate(std::ios_base::failbit);

		}
	}
	return istm;
}
