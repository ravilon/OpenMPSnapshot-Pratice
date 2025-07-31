#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <omp.h>
#include <cmath>
#include <sys/time.h>

using namespace std;

struct RGB
{
    int r; // Vermelho
    int g; // Verde
    int b; // Azul
    RGB(int R, int G, int B) : r(R), g(G), b(B){};
    RGB() : r(0), g(0), b(0){};
};

struct Grey
{
    int value;
    Grey(int v) : value(v){};
    Grey() : value(0){};
};

// Dados da imagem
template <class T>
class Image
{
public:
    string width;             // Largura da imagem
    string height;            // Altura da imagem
    string type;              // Tipo da imagem
    string max_value;         // Máximo valor
    vector<vector<T>> pixels; // Pixels da imagem (Matriz)
    Image(){};
    Image(string w, string h, string t, string max_val) : width(w), height(h), type(t), max_value(max_val){};
};

// Classe de leitura e escrita de imagem
class ImageReaderWriter
{
public:
    Image<RGB> ppmReader(ifstream &inFile)
    {
        string width, height, type, trash, max_value;
        inFile >> type;
        inFile >> ws; // Limpa o buffer
        getline(inFile, trash, '\n');
        inFile >> width >> height >> max_value;

        int w = stoi(width);
        int h = stoi(height);

        Image<RGB> img(width, height, type, max_value);
        for (int i = 0; i < h; i++)
        {
            vector<RGB> row;
            for (int j = 0; j < w; j++)
            {
                string r, g, b;
                if (inFile >> r >> g >> b)
                {
                    RGB pixel(stoi(r), stoi(g), stoi(b));
                    row.push_back(pixel);
                }
                else
                    exit(0);
            }
            img.pixels.push_back(row);
        }
        return img;
    };

    string ppmWriter(Image<RGB> &img)
    {
        string res = "";
        res += img.type + "\n";
        res += img.width + " ";
        res += img.height + "\n";
        res += img.height + "\n";
        for (int i = 0; i < img.pixels.size(); i++)
        {
            string row = "";
            for (int j = 0; j < img.pixels[i].size(); j++)
            {
                row += to_string(img.pixels[i][j].r) + " ";
                row += to_string(img.pixels[i][j].g) + " ";
                row += to_string(img.pixels[i][j].b) + " ";
            }
            res += row + "\n";
        }
        return res;
    }

    string pgmWriter(Image<Grey> &img)
    {
        string res = "";
        res += img.type + "\n";
        res += img.width + " ";
        res += img.height + "\n";
        res += img.max_value + "\n";
        for (int i = 0; i < img.pixels.size(); i++)
        {
            string row = "";
            for (int j = 0; j < img.pixels[i].size(); j++)
            {
                row += to_string(img.pixels[i][j].value) + " ";
            }
            res += row + "\n";
        }
        return res;
    }
};

// Aplica o filtro de identificação de bordas verticais
void processy(const vector<vector<RGB>> chunk, vector<RGB> &pixelout, unsigned colunas)
{
    unsigned i, j, k; // Variáveis axiliares
    int kernel[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    for (k = 1; k < colunas - 1; k++)
    {
        pixelout[k].r = 0;
        for (i = 0; i < 3; i++)
        {
            for (j = 0; j < 3; j++)
            {
                pixelout[k].r += chunk[i][k + j - 1].r * kernel[i][j];
            }
        }
        pixelout[k].g = pixelout[k].r;
        pixelout[k].b = pixelout[k].r;
    }
}

// Aplica o filtro de identificação de bordas horizontais
void processx(const vector<vector<RGB>> chunk, vector<RGB> &pixelout, unsigned colunas)
{
    unsigned i, j, k; // Variáveis axiliares
    int kernel[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

    for (k = 1; k < colunas - 1; k++)
    {
        pixelout[k].r = 0;
        for (i = 0; i < 3; i++)
        {
            for (j = 0; j < 3; j++)
            {
                pixelout[k].r += chunk[i][k + j - 1].r * kernel[i][j];
            }
        }
        pixelout[k].g = pixelout[k].r;
        pixelout[k].b = pixelout[k].r;
    }
}

// Determina o pixel output resultado da aplicação dos filtros
void join(vector<RGB> &pixelinx, vector<RGB> &pixeliny, vector<RGB> &pixelout, unsigned colunas)
{
    unsigned i;
    for (i = 1; i < colunas - 1; i++)
    {
        pixelout[i].r = sqrt(pixelinx[i].r * pixelinx[i].r + pixeliny[i].r * pixeliny[i].r);
        pixelout[i].g = pixelout[i].r;
        pixelout[i].b = pixelout[i].r;
    }
}

// Aplica uma escala de cinzas numa imagem RGB
void RGBtoGrey(const Image<RGB> &ImRGB, Image<Grey> &ImGrey)
{
    ImGrey.height = ImRGB.height;
    ImGrey.width = ImRGB.width;
    ImGrey.type = "P2";
    ImGrey.max_value = ImRGB.max_value;

    ImGrey.pixels.resize(stoi(ImGrey.height));
    for (unsigned i = 0; i < stoi(ImGrey.height); i++)
        ImGrey.pixels[i].resize(stoi(ImGrey.width));

    for (unsigned i = 1; i < stoi(ImRGB.height) - 1; i++)
    {
        for (unsigned j = 1; j < stoi(ImRGB.width) - 1; j++)
        {
            ImGrey.pixels[i][j] = ((0.3 * ImRGB.pixels[i][j].r) + (0.59 * ImRGB.pixels[i][j].g) + (0.11 * ImRGB.pixels[i][j].b));
        }
    }
}

int main(int argc, char *argv[])
{
    struct timeval start, stop;

    string image_in;                      // Imagem de entrada
    int n_threads;                        // Número de threads utilizadas na execução
    unsigned i, j, k, l, colunas, linhas; // Variáveis auxiliares
    vector<vector<RGB>> chunk(3);         // Matriz que armazena pixels vizinhos de uma linha

    // Imagens
    vector<string> images = {"./data/world1.ppm", "./data/world2.ppm", "./data/world3.ppm", "./data/world4.ppm", "./data/Normal2.ppm"};

    if (argc < 3)
    {
        cout << "<Endereco da imagem>" << endl;
        exit(0);
    }

    switch (atoi(argv[2]))
    {
    case 1:
        image_in = images[0];
        break;
    case 2:
        image_in = images[1];
        break;
    case 3:
        image_in = images[2];
        break;
    case 4:
        image_in = images[3];
        break;
    case 5:
        image_in = images[4];
        break;
    default:
        break;
    }

    n_threads = atoi(argv[1]);

    ifstream inFile;
    string file_path = image_in;

    inFile.open(file_path, ifstream::in);
    if (!inFile.is_open())
    {
        cout << "Falha ao abrir a imagem " << file_path << endl;
        exit(0);
    }

    ImageReaderWriter img_rw;
    Image<RGB> myImage = img_rw.ppmReader(inFile);
    Image<RGB> outImagex = myImage;
    Image<RGB> outImagey = myImage;
    Image<RGB> outImage = myImage;
    Image<Grey> outImageGrey;

    inFile.close();

    colunas = stoi(myImage.width);
    linhas = stoi(myImage.height);

    for (i = 0; i < 3; i++)
    {
        chunk[i].resize(colunas);
    }

    gettimeofday(&start, 0);

    #pragma omp parallel num_threads(n_threads) default(none) firstprivate(chunk) shared(myImage, outImagex, outImagey, outImage, colunas, linhas)
    {
        #pragma omp for schedule(dynamic)
        for (int i = 1; i < linhas - 1; i++)
        {

            for (int k = i - 1; k <= i + 1; k++)
            {
                for (int l = 0; l < colunas; l++)
                {
                    chunk[k - i + 1][l] = myImage.pixels[k][l];
                }
            }
            {
                processx(chunk, outImagex.pixels[i], colunas);
                processy(chunk, outImagey.pixels[i], colunas);
            }
            {
                join(outImagex.pixels[i], outImagey.pixels[i], outImage.pixels[i], colunas);
            }
        }
    }

    gettimeofday(&stop, 0);

    // Criação de imagens output
    {
        string img_streamx = img_rw.ppmWriter(outImagex);
        string img_streamy = img_rw.ppmWriter(outImagey);
        string img_stream = img_rw.ppmWriter(outImage);

        RGBtoGrey(outImage, outImageGrey);
        string img_streamGrey = img_rw.pgmWriter(outImageGrey);

        ofstream outFilex("./paralelo/out/outputx.ppm", ofstream::out);
        if (outFilex.is_open())
        {
            outFilex << img_streamx;
        }
        ofstream outFiley("./paralelo/out/outputy.ppm", ofstream::out);
        if (outFiley.is_open())
        {
            outFiley << img_streamy;
        }
        ofstream outFile("./paralelo/out/output.ppm", ofstream::out);
        if (outFile.is_open())
        {
            outFile << img_stream;
        }
        ofstream outFileGrey("./paralelo/out/outputGrey.pgm", ofstream::out);
        if (outFileGrey.is_open())
        {
            outFileGrey << img_streamGrey;
        }
        cout << "Imagem processada" << endl;

        inFile.close();
        outFilex.close();
        outFiley.close();
        outFile.close();
        outFileGrey.close();
    }

    // Impressão de resultados em arquivo
    {
        FILE *fp;
        char outputFilename[] = "./paralelo/tempo_omp_filters.txt";

        fp = fopen(outputFilename, "a");
        if (fp == NULL)
        {
            fprintf(stderr, "Nao foi possivel abrir o arquivo %s!\n", outputFilename);
            exit(1);
        }

        fprintf(fp,
                "\tTempo: %1.2e \n",
                ((double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec)));
        fclose(fp);
    }
    return 0;
}