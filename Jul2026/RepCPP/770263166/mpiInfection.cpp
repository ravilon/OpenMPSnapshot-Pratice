#include <thread> //used for getting the number of cpu cores available (this is the hardware core count, not the slurm count)
#include <omp.h> //openMP library
#include <mpi.h>//MPI library
#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
using namespace std;
using namespace std::chrono;
#define ROOT_NODE 0

/**
 * @brief A class representing an individual that can either be
 * - infected
 * - immune
 * - susceptible
 * 
 */
class Person {
    private:
        float x; //x coordinate
        float y; //y coordinate
        bool immuneStatus; //if the individual is immune
        bool infectedStatus; //if the individual is infected
        int timeInfected; //The amount of time the individual has been infected
        int timeImmune; //The amount of time the individual has been immune
        bool alive; //If the individual is alive
        int infectionDuration; //The duration of infection
        int immunityDuration; //The duration of immunity
    public:
        Person() {};
        /**
         * @brief Construct a new Person object with specified initial conditions
         * 
         * @param initialX the initial x coordinate
         * @param initialY the initial y coordinate
         * @param initialImmune if they start off immune
         * @param initialInfected if they start off infected
         * @param infectionTime the duration of infection
         * @param immuneTime the duration of immunity
         */
        Person(float initialX, float initialY, bool initialImmune, bool initialInfected, int infectionTime, int immuneTime) {
            x = initialX;
            y = initialY;
            immuneStatus = initialImmune;
            infectedStatus = initialInfected;
            timeInfected = 0;
            timeImmune = 0;
            infectionDuration = infectionTime;
            immunityDuration = immuneTime;
            alive = true;
        }

        /**
         * @brief Return the x coordinate of the Person
         * 
         * @return float the x coordinate
         */
        float getX() {
            return x;
        }

        /**
         * @brief Return the y coordiante of the Person
         * 
         * @return float the y coordinate
         */
        float getY() {
            return y;
        }

        /**
         * @brief Return a status indicating whether the person is immune or not
         * 
         * @return true if the individual is immune
         * @return false if the individual is not immune
         */
        bool isImmune() {
            return immuneStatus;
        }

        /**
         * @brief Return a status indicating whether a person is infected or not
         * 
         * @return true if the individual is infected
         * @return false if the individual is not infected
         */
        bool isInfected() {
            return infectedStatus;
        }

        /**
         * @brief Set a new x coordinate for the Person
         * 
         * @param newX the new x coordinate
         */
        void setX(float newX) {
            x = newX;
        }

        /**
         * @brief Set a new y coordinate for the Person
         * 
         * @param newY the new y coordinate
         */
        void setY(float newY) {
            y = newY;
        }

        /**
         * @brief Infect the current person
         * 
         */
        void infect() {
            infectedStatus = true;
        }

        /**
         * @brief Immunise the current person
         * 
         */
        void immunise() {
            immuneStatus = true;
        }

        /**
         * @brief If the individual is infected, increase their time. If this passes the threshold, then
         * remove their infected status, and check whether they die.
         * 
         * @param deathRate the percentage chance of the individual dying
         */
        void increaseTimeInfected(float deathRate) {
            timeInfected += 1;
            if (timeInfected == infectionDuration) {
                infectedStatus = false;
                timeInfected = 0;
                immuneStatus = true;
                float random = ((float)rand()/(float)RAND_MAX);
                if (random < deathRate) {
                    alive = false;
                }
            }
        }

        /**
         * @brief If the indiviual is immune, increase their time immune. If this passes the threshold, then
         * remove their immune status.
         * 
         */
        void increaseTimeImmune() {
            timeImmune += 1;
            if (timeImmune == immunityDuration) {
                immuneStatus = false;
                timeImmune = 0;
            }
        }

        /**
         * @brief Return whether the individual is alive or not
         * 
         * @return true if the individual is alive
         * @return false if the individual is dead
         */
        bool isAlive() {
            return alive;
        }
};

/**
 * @brief A class which represents a world in which people live
 * 
 */
class World {
    private:
        int xmin; //minimum x bound
        int xmax; //maximum x bound
        int ymin; //minimum y bound
        int ymax; //maximum y bound
        int numSusceptible; //The number of susceptible individuals
        int numInfected; //The number of infected individuals
        int total; //The total number of individuals
        std::vector<Person> people; // The people in the World
        float infectionRate; //Infection Spread Rate
        float deathRate; //Infection Death Rate
    public:
        World() {};

        /**
         * @brief Construct a new World object in which Person's live and can simulate infection spread.
         * 
         * @param x1 the minimum x bound
         * @param x2 the maximum x bound
         * @param y1 the minimum y bound
         * @param y2 the maximum y bound
         * @param initialSusceptible the initial number of susceptible people
         * @param initialInfected the initial number of infected people
         * @param infection the infection rate
         * @param death the death rate
         * @param infectionTime how long infection lasts
         * @param immuneTime how long immunity lasts
         */
        World(float x1, float x2, float y1, float y2, int initialSusceptible, int initialInfected, float infection, float death,
        int infectionTime, int immuneTime) {
            xmin = x1;
            xmax = x2;
            ymin = y1;
            ymax = y2;
            numSusceptible = initialSusceptible;
            numInfected = initialInfected;
            total = numSusceptible + numInfected;
            infectionRate = infection;
            deathRate = death;
            people.reserve(total);
            for (int i=0; i < numSusceptible; i++) {
                people.push_back(Person(rand() % xmax, rand() % ymax, false, false, infectionTime, immuneTime));
            }
            for (int i = numSusceptible; i < total; i++) {
                people.push_back(Person(rand() % xmax, rand() % ymax, false, true, infectionTime, immuneTime));
            }
        }

        /**
         * @brief Initiates one tick of time in the world, where people move and can infect, die, become immune, move etc
         * 
         */
        void tick() {
            int num; 
            float random;
            float distance;
            float xDistance;
            float yDistance;
            float personX;
            float personY;

            int rank, size;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            //Deine variables that can divide the array into equal parts for each node
            int local_total = total / size;
            int start = rank * local_total;
            int end = (rank + 1) * local_total;

            //For each person, if they are alive, and if they are not infected or immune, then check nearby infected individuals
            #pragma omp parallel for
            for (int i = start; i < end; i++) {
                if (people[i].isAlive()) {
                    if (!(people[i].isInfected() || people[i].isImmune())) {
                        personX = people[i].getX();
                        personY = people[i].getY();
                        for (int j = 0; j < total; j++) {
                            //If the person is infected
                            if (people[j].isInfected()) {
                                //Calculate distance from the original person and see if they are close enough
                                xDistance = personX - people[j].getX();
                                yDistance = personY - people[j].getY();
                                distance = sqrt( xDistance * xDistance + yDistance * yDistance );
                                if (distance < 20) {
                                    //Calculate from infection rate if the infection is transmitted
                                    random =((float)rand()/(float)RAND_MAX);
                                    if (random < infectionRate) {
                                        people[i].infect(); 
                                    }
                                }
                            }
                        }
                    }
                    //Move people randomly
                    num = rand() % 3 - 1;
                    people[i].setX( people[i].getX() + num );
                    num = rand() % 3 - 1;
                    people[i].setY( people[i].getY() + num );
                    // If the person stepped out of bounds, wrap them around accordingly
                    if (people[i].getX() < xmin) {
                        people[i].setX(people[i].getX() + xmax);
                    } else if (people[i].getX() > xmax) {
                        people[i].setX(people[i].getX() - xmax );
                    }

                    if (people[i].getY() < ymin) {
                        people[i].setY(people[i].getY() + ymax);
                    } else if (people[i].getY() > ymax) {
                        people[i].setY(people[i].getY() - ymax);
                    }
                    if (people[i].isInfected()) {
                        people[i].increaseTimeInfected(deathRate);
                    } else if (people[i].isImmune()) {
                        people[i].increaseTimeImmune();
                    }
            }
        }
        
    }
        
        void writeToFile(ofstream& file) {
            if (file.is_open()) {
                for (Person person : people) {
                    file << person.getX() << " " << person.getY() << " " << person.isImmune() << " " << person.isInfected() << " " << person.isAlive() << ",";
                    file << "\n";
                }
            }
        }
};

int main() {
    //Initialise the MPI
    MPI_Init(NULL, NULL);
    auto start = high_resolution_clock::now();
    World world = World(0, 2500, 0, 2500, 900, 100, 1.0, 0.2, 300, 300);
    ofstream fw("OutputDataMPI.txt", std::ofstream::out);
    for (int i = 0; i < 1; i++) {
        world.tick();
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    fw << duration.count() << "\n" << endl;
    world.writeToFile(fw);
    fw.close();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
