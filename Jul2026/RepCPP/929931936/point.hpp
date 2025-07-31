/*
   Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
 */

#pragma once

namespace hippoLBM
{

  typedef std::array<int,3> int3d;

  inline
    ONIKA_HOST_DEVICE_FUNC int3d operator+(int3d& a, int b)
    {
      int3d res;
      for (int dim = 0 ; dim < 3 ; dim++) res[dim] = a[dim] + b;
      return res;
    }

  // should change it to n-DIM
  template<int DIM>
    struct point
    {
      int3d position;

      ONIKA_HOST_DEVICE_FUNC point() {};
      ONIKA_HOST_DEVICE_FUNC point(int x, int y, int z) { position[0] = x; position[1] = y; position[2] = z; }
      ONIKA_HOST_DEVICE_FUNC inline int get_val(int dim) {return position[dim];}
      ONIKA_HOST_DEVICE_FUNC inline void set_val(int dim, int val) { position[dim] = val;}
      ONIKA_HOST_DEVICE_FUNC inline int& operator[](int dim) {return position[dim];}
      ONIKA_HOST_DEVICE_FUNC inline const int& operator[](int dim) const {return position[dim];}  
      void print() 
      {
        for(int dim = 0; dim < DIM ; dim++) 
        {
          onika::lout << " " << position[dim];
        }

        onika::lout << std::endl;
      }

      ONIKA_HOST_DEVICE_FUNC point<DIM> operator+(point<DIM>& p)
      {
        point<DIM> res = {position[0] + p[0], position[1] + p[1], position[2] + p[2]};
        return res;
      } 

      ONIKA_HOST_DEVICE_FUNC point<DIM> operator-(point<DIM>& p)
      {
        point<DIM> res = {position[0] - p[0], position[1] - p[1], position[2] - p[2]};
        return res;
      } 
    };

  template<int DIM>
    ONIKA_HOST_DEVICE_FUNC point<DIM> min(point<DIM>& a, point<DIM>& b)
    {
      point<DIM> res;
      for(int dim = 0 ; dim < DIM ; dim++)
      {
        res[dim] = std::min(a[dim], b[dim]);
      }
      return res;
    }

  template<int DIM>
    ONIKA_HOST_DEVICE_FUNC point<DIM> max(point<DIM>& a, point<DIM>& b)
    {
      point<DIM> res;
      for(int dim = 0 ; dim < DIM ; dim++)
      {
        res[dim] = std::max(a[dim], b[dim]);
        //std::cout << " res " << res[dim] << " max( " << a[dim] << " , " << b[dim] << ")" << std::endl;
      }
      return res;
    }
}
