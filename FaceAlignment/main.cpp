#include <cstdio>
#include "../Lib3000fps/lbf_common.hpp"

using namespace std;
using namespace lbf;

// dirty but works
int train(int);
int test(void);
int prepare(void);
int run(void);


int main(int argc, char **argv) {
    if (argc != 2) {
        LOG("We need an argument");
    }
    else if (strcmp(argv[1], "train") == 0) {
        train(0);
    }
    else if (strcmp(argv[1], "resume") == 0) {
        int start_from;
        printf("Which stage you want to resume from: ");
        scanf("%d", &start_from);
        train(start_from);
    }
    else if (strcmp(argv[1], "test") == 0) {
        test();
    }
    else if (strcmp(argv[1], "prepare") == 0) {
        prepare();
    }
    else if (strcmp(argv[1], "run") == 0) {
        run();
    }
    else {
        LOG("Wrong Arguments.");
    }
	system("pause");
    return 0;
}