#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s file zero_from\n", argv[0]); return 2; }
    int zero_from = atoi(argv[2]);
    if (zero_from < 0 || zero_from > 508) return 2;
    int fd = open(argv[1], O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) || st.st_size < 512) return 2;
    size_t size = (size_t)st.st_size;
    const uint8_t *d = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (d == MAP_FAILED) return 2;
    const uint8_t initial_key[16] = {
        0x06,0x0e,0x2b,0x34,0x02,0x05,0x01,0x01,
        0x0d,0x01,0x02,0x01,0x01,0x02,0x04,0x00
    };
    uint8_t pad[512], known[512] = {0};
    for (size_t i = 0; i < 16; i++) { pad[i] = d[i] ^ initial_key[i]; known[i] = 1; }
    for (size_t i = zero_from; i < 512; i++) { pad[i] = d[i]; known[i] = 1; }
    const uint8_t klv[4] = {0x06, 0x0e, 0x2b, 0x34};
    size_t count = 0;
    for (size_t o = zero_from; o + 4 <= size; o++) {
        size_t p = o & 511;
        if (!known[p] || !known[p+1] || !known[p+2] || !known[p+3]) continue;
        if ((uint8_t)(d[o] ^ pad[p]) != klv[0]) continue;
        if ((uint8_t)(d[o+1] ^ pad[p+1]) != klv[1]) continue;
        if ((uint8_t)(d[o+2] ^ pad[p+2]) != klv[2]) continue;
        if ((uint8_t)(d[o+3] ^ pad[p+3]) != klv[3]) continue;
        printf("0x%08zx phase=%zu\n", o, p);
        count++;
    }
    printf("hits=%zu\n", count);
    munmap((void *)d, size); close(fd);
    return 0;
}
