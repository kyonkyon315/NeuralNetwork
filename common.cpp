#include "common.h"
//https://www.youtube.com/@ProgrammerCpp �l����̈��p

std::istream& operator>>(std::istream& istm, char&& r) {
	char ch;
	if (isgraph(r) ? istm >> ch : istm.read(&ch, 1)) {
		//�������ǂ߂�
		if (ch != r) {
			//�ُ�i�ʂ̕����������j
			istm.setstate(std::ios_base::failbit);

		}
	}
	return istm;
}
