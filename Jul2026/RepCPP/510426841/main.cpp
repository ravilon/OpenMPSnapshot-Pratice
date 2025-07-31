#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
#include <algorithm>
#include <fstream>
#include <random>
#include <time.h>
#include <fcntl.h>
#include <ctime>
#include <omp.h>
#include <sys/time.h>

int main(int argc, char *argv[])
{
    struct timeval _ttime;
    struct timezone _tzone;

    gettimeofday(&_ttime, &_tzone);
    double time_start = (double)_ttime.tv_sec + (double)_ttime.tv_usec / 1000000.;

    // Unused input arguments
    if (argc > 0)
    {
        for (int i = 0; i < argc; ++i)
        {
            printf("%s\n", argv[i]);
        }
    }

    // print the argument count
    printf("argc: %d\n", argc);

    // if there are 7 input arguments, then we try to parse them as integers and user them as the users numbers.
    int user_numbers[7] = {4, 7, 17, 20, 32, 42, 14};
    int best_attempt[7] = {0, 0, 0, 0, 0, 0, 0};
    int best_attempt_correct_rows[7] = {0, 0, 0, 0, 0, 0, 0};

    if (argc == 8)
    {
        for (int i = 0; i < 7; ++i)
        {
            user_numbers[i] = atoi(argv[i + 1]);
        }
    }
    else
    {
        printf("No input arguments\n");
    }

    // run forever untill the user has won the lottery
    bool has_won = false;
    unsigned long long int attempts = 0ULL;
    int tid;
    int maximum_correct_numbers = 0;

    while (!has_won)
#pragma omp parallel shared(user_numbers, has_won, maximum_correct_numbers, best_attempt) private(tid)
    {
#pragma omp atomic
        attempts++;

        tid = omp_get_thread_num();

        // lottetry game, let the user input 7 numbers
        srand(time(NULL));

        // check if the user_numbers array is all zeroes
        bool all_zeroes = true;
        for (int i = 0; i < 7; ++i)
        {
            if (user_numbers[i] != 0)
            {
                all_zeroes = false;
            }
        }

        if (all_zeroes)
        {
            // read the numbers from stdin, only in the range of 1-49 and no duplicates with the title "please choose 7 numbers in the range of 1-49".
            for (int i = 0; i < 7; ++i)
            {
                printf("please choose %d number in the range of 1-49: ", i + 1);
                scanf("%d", &user_numbers[i]);
                while (user_numbers[i] < 1 || user_numbers[i] > 49)
                {
                    printf("please choose %d number in the range of 1-49: ", i + 1);
                    scanf("%d", &user_numbers[i]);
                }
                for (int j = 0; j < i; ++j)
                {
                    if (user_numbers[i] == user_numbers[j])
                    {
                        printf("please choose %d number in the range of 1-49: ", i + 1);
                        scanf("%d", &user_numbers[i]);
                    }
                }
            }
        }

        // draw 7 random numbers between 1 and 49, no duplicates and compare them to the users input. If a number matches output it to stdout in bold green. if a number is not in the input, output it in bold red. Output the winning row separately in bold yellow with the title "winnin row".
        int random_numbers[7];
        for (int i = 0; i < 7; ++i)
        {
            random_numbers[i] = rand() % 49 + 1;
            for (int j = 0; j < i; ++j)
            {
                if (random_numbers[i] == random_numbers[j])
                {
                    random_numbers[i] = rand() % 49 + 1;
                    j = -1;
                }
            }
        }

        int correct = 0;
        for (int i = 0; i < 7; ++i)
        {
            if (std::find(user_numbers, user_numbers + 7, random_numbers[i]) != user_numbers + 7)
            {
                // printf("\033[1;32m%d\033[0m ", random_numbers[i]);
                correct++;
            }
            else
            {
                // printf("\033[1;31m%d\033[0m ", random_numbers[i]);
            }
        }

        if (correct > maximum_correct_numbers)
        {
            maximum_correct_numbers = correct;

            // insert the user_numbers into the best_attempt
            for (int i = 0; i < 7; ++i)
            {
                best_attempt[i] = user_numbers[i];

                best_attempt_correct_rows[i] = random_numbers[i];
            }
        }

        // if the user has won, print the title "you have won" and the winning numbers in bold green.
        if (std::equal(user_numbers, user_numbers + 7, random_numbers))
        {
            printf("\n\n\033[1;32myou have won!\033[0m\n");
            has_won = true;
        }

        if (attempts % 100000ULL == 0)
        {
            // Only master thread does this
            if (tid == 0)
            {
                // print the maximum_correct_numbers in blue
                printf("\n\n\033[1;34mmaximum correct numbers: \033[0m%d\n", maximum_correct_numbers);

                // print the best_attempt in green where the number are in the best_attempt_correct_rows else in red
                printf("\n\n\033[1;34mbest attempt: \033[0m");
                for (int i = 0; i < 7; ++i)
                {
                    if (std::find(best_attempt_correct_rows, best_attempt_correct_rows + 7, best_attempt[i]) != best_attempt_correct_rows + 7)
                    {
                        printf("\033[1;32m%d\033[0m ", best_attempt[i]);
                    }
                    else
                    {
                        printf("\033[1;31m%d\033[0m ", best_attempt[i]);
                    }
                }

                // print the winning numbers in bold yellow with the title "winning row"
                printf("\n\n\033[1;33mwinning row:\033[0m \n");
                for (int i = 0; i < 7; ++i)
                {
                    printf("\033[1;33m%d\033[0m ", best_attempt_correct_rows[i]);
                }

                // print the number of attempts
                printf("\n\n\nattempts: %llu\n", attempts);

                gettimeofday(&_ttime, &_tzone);
                double time_end = (double)_ttime.tv_sec + (double)_ttime.tv_usec / 1000000.;

                printf("   Wall clock run time    = %.1lf secs\n", time_end - time_start);
            }
        }
    } // end of while

    return (0);
}
