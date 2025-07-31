#pragma once


#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
namespace py = pybind11;


#include "datatypes/vector.hpp"
#include "accelerator/accelerator.hpp"


namespace paracabs
{
    namespace datatypes
    {
        /// MatrixTP: a thread private 2-index data structure
        /////////////////////////////////////////////////////
        template <typename type, typename XThreads>
        struct MatrixTP : public Vector<type>, XThreads
        {
            size_t nrows = 0;
            size_t ncols = 0;


            ///  Constructor (no argument)
            //////////////////////////////
            inline MatrixTP ()
            {
                Vector<type>::set_dat ();
            }

            ///  Copy constructor (shallow copy)
            ////////////////////////////////////
            inline MatrixTP (const MatrixTP& m)
            {
                nrows    = m.nrows;
                ncols    = m.ncols;

                Vector<type>::ptr            = m.ptr;
                Vector<type>::allocated      = false;
                Vector<type>::allocated_size = 0;
                Vector<type>::set_dat ();
            }

            ///  Constructor (double argument)
            //////////////////////////////////
            inline MatrixTP (const size_t nr, const size_t nc)
            {
                MatrixTP<type, XThreads>::resize (nr, nc);
            }

            ///  Resizing both the std::vector and the allocated memory
            ///    @param[in] size : new size for std::vector
            ///////////////////////////////////////////////////////////
            inline void resize (const size_t nr, const size_t nc)
            {
                nrows = nr;
                ncols = nc;

                Vector<type>::vec.resize (nrows*ncols*XThreads::tot_nthreads());
                Vector<type>::copy_vec_to_ptr ();
                Vector<type>::set_dat ();
            }

            ///  Row major indexing function
            ////////////////////////////////
            accel inline size_t index (const size_t t, const size_t id_r, const size_t id_c) const
            {
                return id_c + ncols*(id_r + nrows*t);
            }

            ///  Access operators
            accel inline type  operator() (const size_t id_r, const size_t id_c) const
            {
                return Vector<type>::dat[index(XThreads::thread_id(), id_r, id_c)];
            }

            accel inline type &operator() (const size_t id_r, const size_t id_c)
            {
                return Vector<type>::dat[index(XThreads::thread_id(), id_r, id_c)];
            }

            accel inline type  operator() (const size_t t, const size_t id_r, const size_t id_c) const
            {
                return Vector<type>::dat[index(t, id_r, id_c)];
            }

            accel inline type &operator() (const size_t t, const size_t id_r, const size_t id_c)
            {
                return Vector<type>::dat[index(t, id_r, id_c)];
            }
        };
    }
}
