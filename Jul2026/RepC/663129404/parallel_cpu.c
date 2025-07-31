#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <omp.h>
#include <time.h>

#include "settings.h"

char folder[100];
char print = 0;

int calculate_weight(char stra[], char strb[])
{
    int len = strlen(stra);
    char *a;
    char *b;
    a = malloc(len);
    b = malloc(len);
    memcpy(a, stra, len);
    memcpy(b, strb, len);
    a[len] = '\0';
    b[len] = '\0';
    // printf("%s %s %d\n", &a[0], &b[0], len);
    for (int i = 1; i < len; i++)
    {
        b[len - i] = '\0';
        if (strcmp(&a[i], &b[0]) == 0)
        {
            free(a);
            free(b);
            return i * i;
        }
    }
    free(a);
    free(b);
    return -1;
}

void main_func(char filename[])
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Failed to open file: %s\n", filename);
        return;
    }
    char dna[MAX_DNA_LEN][MAX_WORD_LEN];
    int dna_count = 0;

    char line[MAX_WORD_LEN];

    while (fgets(line, sizeof(line), file))
    {
        line[strcspn(line, "\n")] = '\0';
        strcpy(dna[dna_count], line);
        dna_count++;
    }

    int *neighbours;
    neighbours = malloc((dna_count * dna_count) * sizeof(int));
#pragma omp parallel for
    for (int i = 0; i < dna_count; i++)
    {
        // int id
        for (int j = 0; j < dna_count; j++)
        {
            neighbours[i * dna_count + j] = calculate_weight(dna[i], dna[j]);
        }
    }

    /*
    for (int i = 0; i < dna_count; i++)
        printf("%s\n", dna[i]);
    */

    if (print)
    {
        for (int i = 0; i < dna_count; i++)
        {
            for (int j = 0; j < dna_count; j++)
            {
                printf("%d ", neighbours[i * dna_count + j]);
            }
            printf("\n");
        }
    }
    free(neighbours);

    fclose(file);
}
int main(int argc, char *argv[])
{
    for (int i = 0; i < argc; i++)
    {
        if (i == 1)
        {
            strcpy(folder, argv[i]);
        }
        else if (i == 2)
        {
            print = atoi(argv[i]);
        }
    }
    const char *folders[] = {folder};
    int num_folders = sizeof(folders) / sizeof(folders[0]);

    for (int i = 0; i < num_folders; i++)
    {
        const char *ins_dir = folders[i];
        DIR *dir = opendir(ins_dir);
        if (dir == NULL)
        {
            printf("Failed to open directory: %s\n", ins_dir);
            continue;
        }

        struct dirent *entry;
        char filename[200];
        double begin;
        double end;
        double time_spent;
        while ((entry = readdir(dir)) != NULL)
        {
            if (entry->d_type == DT_REG)
            {

                if (print)
                {
                    printf("%s\n", entry->d_name);
                }

                begin = omp_get_wtime();
                snprintf(filename, sizeof(filename), "%s/%s", ins_dir, entry->d_name);

                main_func(filename);

                end = omp_get_wtime();
                time_spent = end - begin;

                printf("%f\n", time_spent * 1000);
            }
        }

        closedir(dir);
    }

    return 0;
}
