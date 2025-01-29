// add.c

// Export the function for shared libraries
__attribute__((visibility("default")))
int add_my_nums(int x, int y) {
    return x + y;
}
