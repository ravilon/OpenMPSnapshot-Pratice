#pragma once
#include "PCH.hpp"

/**
 *  @struct MATRIX_INFO
    @brief struct which contains
    @param mat_name std::string - Matrix name
    @param len_row  uint32_t - Row length
    @param len_col  uint32_t - Column length
    @param num_nz   uint32_t - Number of none zero
*/
struct MATRIX_INFO
{
public:
    std::string matrix_name;
    uint32_t num_row;
    uint32_t num_col;
    uint32_t num_nz;

    bool operator==(const MATRIX_INFO &target)
    {
        return comp(target);
    }

    bool operator!=(const MATRIX_INFO &target)
    {
        return !comp(target);
    }

private:
    bool comp(const MATRIX_INFO &target)
    {
        if (
            (this->num_row != target.num_row) ||
            (this->num_col != target.num_col) ||
            (this->num_nz != target.num_nz))
        {
            return false;
        }
        return true;
    }
};

/**
 *  @struct D_MATRIX
    @brief struct which contains dense matrix information
    @param mat_data MATRIX_INFO - Matrix information
    @param matrix std::vector<std::vector<double>> - vector of vector with matrix elements
*/
struct D_MATRIX
{
public:
    MATRIX_INFO mat_data;
    std::vector<std::vector<double>> matrix;

    bool operator==(const D_MATRIX &target)
    {
        return comp(target);
    }

    bool operator!=(const D_MATRIX &target)
    {
        return !comp(target);
    }

private:
    bool comp(const D_MATRIX &target)
    {
        if (this->mat_data != target.mat_data)
        {
            return false;
        }

        for (uint32_t row = 0; row < this->mat_data.num_row; row++)
        {
            if (
                std::equal(
                    this->matrix[row].begin(), this->matrix[row].end(),
                    target.matrix[row].begin(), target.matrix[row].end()) == false)
            {
                return false;
            }
        }
        return true;
    }
};

/**
    @struct MAT_ELE_DATA
    @brief Element storage format for COO format sparse matrix
    @param row_idx    uint32_t - Row idx
    @param col_idx    uint32_t - Column idx
    @param val          double - Value of the coordinate
*/
struct MAT_ELE_DATA
{
public:
    uint32_t row_idx;
    uint32_t col_idx;
    double val;

    bool operator==(const MAT_ELE_DATA &target)
    {
        return comp(target);
    }

    bool operator!=(const MAT_ELE_DATA &target)
    {
        return !comp(target);
    }

private:
    bool comp(const MAT_ELE_DATA &target)
    {
        if (
            (this->row_idx != target.row_idx) ||
            (this->col_idx != target.col_idx) ||
            (this->val != target.val))
        {
            return false;
        }
        return true;
    }
};

/**
    @struct IDX_VAL
    @brief Storage format for sparse matrix CSC/CSR and sparse vector.
    @param idx uint32_t - idx of row(for CSC), colum(for CSR) and value(for sparse vector).
    @param val double - value at that idx
*/
struct IDX_VAL
{
public:
    uint32_t idx;
    double val;

    bool operator==(const IDX_VAL &target)
    {
        return comp(target);
    }

    bool operator!=(const IDX_VAL &target)
    {
        return !comp(target);
    }

private:
    bool comp(const IDX_VAL &target)
    {
        if (
            (this->idx != target.idx) ||
            (this->val != target.val))
        {
            return false;
        }
        return true;
    }
};

/**
    @struct COO
    @brief Coodinate (in Triplet) format sparse matrix
    @param mat_data     MATRIX_INFO - Matrix information
    @param mat_elements vector<MAT_ELE_DATA> - vector of MAT_ELE_DATA
*/
struct COO
{
public:
    MATRIX_INFO mat_data;
    std::vector<MAT_ELE_DATA> mat_elements;

    bool operator==(const COO &target)
    {
        return comp(target);
    }

    bool operator!=(const COO &target)
    {
        return !comp(target);
    }

private:
    bool comp(const COO &target)
    {
        if (this->mat_data != target.mat_data)
        {
            return false;
        }

        if (std::equal(
                this->mat_elements.begin(), this->mat_elements.end(),
                target.mat_elements.begin(), target.mat_elements.end()) == false)
        {
            return false;
        }

        return true;
    }
};

/**
    @brief struct - Compressed sparse column format sparse matrix
    @param mat_data MATRIX_INFO - Matrix information
    @param col_ptr std::vector<uint32_t> - vector of column pointer
    @param row_with_val std::vector<IDX_VAL> - vector of IDX_VAL contains row idx and value
*/
struct CSC
{
    MATRIX_INFO mat_data;
    std::vector<uint32_t> col_ptr;
    std::vector<IDX_VAL> row_and_val;

    bool operator==(const CSC &target)
    {
        return comp(target);
    }

    bool operator!=(const CSC &target)
    {
        return !comp(target);
    }

private:
    bool comp(const CSC &target)
    {
        if (this->mat_data != target.mat_data)
            return false;

        if (std::equal(
                this->col_ptr.begin(), this->col_ptr.end(),
                target.col_ptr.begin(), target.col_ptr.end()) == false)
        {
            return false;
        }

        if (std::equal(
                this->row_and_val.begin(), this->row_and_val.end(),
                target.row_and_val.begin(), target.row_and_val.end()) == false)
        {
            return false;
        }

        return true;
    }
};

/**
    @brief struct - Compressed sparse row format sparse matrix
    @param mat_data MATRIX_INFO - Matrix information
    @param row_ptr std::vector<uint32_t> - vector of row pointer
    @param col_with_val std::vector<IDX_VAL> - vector of IDX_VAL contains column idx and value
*/
struct CSR
{
public:
    MATRIX_INFO mat_data;
    std::vector<uint32_t> row_ptr;
    std::vector<IDX_VAL> col_and_val;

    bool operator==(const CSR &target)
    {
        return comp(target);
    }

    bool operator!=(const CSR &target)
    {
        return !comp(target);
    }

private:
    bool comp(const CSR &target)
    {
        if (this->mat_data != target.mat_data)
            return false;

        if (std::equal(
                this->row_ptr.begin(), this->row_ptr.end(),
                target.row_ptr.begin(), target.row_ptr.end()) == false)
        {
            return false;
        }

        if (std::equal(
                this->col_and_val.begin(), this->col_and_val.end(),
                target.col_and_val.begin(), target.col_and_val.end()) == false)
        {
            return false;
        }

        return true;
    }
};

/**The following two struct is designed to store vector information**/
/**
 *  @struct VECTOR_INFO
    @brief struct which contains input vector information
    @param vec_name std::string - Vector name
    @param len_vec  uint32_t - Vector length
    @param num_nz   uint32_t - Number of none zero
*/
struct VECTOR_INFO
{
public:
    std::string vec_name;
    uint32_t len_vec;
    uint32_t num_nz;

    bool operator==(const VECTOR_INFO &target)
    {
        return comp(target);
    }

    bool operator!=(const VECTOR_INFO &target)
    {
        return !comp(target);
    }

private:
    bool comp(const VECTOR_INFO &target)
    {
        if (
            (this->len_vec != target.len_vec) ||
            (this->num_nz != target.num_nz))
        {
            return false;
        }
        return true;
    }
};

/**
 *  @struct D_VECTOR
    @brief struct which contains dense vector information
    @param vec_data VECTOR_INFO - vector information
    @param vec_elements std::vector<double> - vector of value of the vector
*/
struct D_VECTOR
{
public:
    VECTOR_INFO vec_data;
    std::vector<double> vec_element;

    bool operator==(const D_VECTOR &target)
    {
        return comp(target);
    }

    bool operator!=(const D_VECTOR &target)
    {
        return !comp(target);
    }

private:
    bool comp(const D_VECTOR &target)
    {
        if (this->vec_data != target.vec_data)
            return false;

        if (std::equal(
                this->vec_element.begin(), this->vec_element.end(),
                target.vec_element.begin(), target.vec_element.end()) == false)
        {
            return false;
        }
        return true;
    }
};

/**
 *  @struct S_VECTOR
    @brief struct which contains sparse vector information
    @param vec_data VECTOR_INFO - vector information
    @param vec_elements std::vector<double> - vector of vector value
*/
struct S_VECTOR
{
public:
    VECTOR_INFO vec_data;
    std::vector<IDX_VAL> vec_element;

    bool operator==(const S_VECTOR &target)
    {
        return comp(target);
    }

    bool operator!=(const S_VECTOR &target)
    {
        return !comp(target);
    }

private:
    bool comp(const S_VECTOR &target)
    {
        if (this->vec_data != target.vec_data)
            return false;

        if (std::equal(
                this->vec_element.begin(), this->vec_element.end(),
                target.vec_element.begin(), target.vec_element.end()) == false)
        {
            return false;
        }

        return true;
    }
};
