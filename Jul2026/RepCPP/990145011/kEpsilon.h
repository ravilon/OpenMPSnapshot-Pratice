// =================================================================================================
// ====================================================================================== kEpsilon.h
//
// Condiciones de contorno de tipo "wall function" para el modelo k-ε
//
// ================================================================= Copyright © 2025 Alberto Escrig
// ======================================================================================== INCLUDES

#pragma once

import VF;
import std;

namespace VF
{
// ============================================================================== VARIABLES EN LÍNEA
// =================================================================================================

inline constexpr double C1 = 1.44;
inline constexpr double C2 = 1.92;
inline constexpr double σk = 1.0;
inline constexpr double σε = 1.3;
inline constexpr double Cμ = 0.09;
inline constexpr double κ  = 0.41;
inline constexpr double E  = 9.8;

// =========================================================================== DECLARACIÓN DE CLASES
// ====================================================================================== TkWallFunc

template<std::size_t d, std::size_t r> requires (r == 0u)
class TkWallFunc : public TNeumann<d, 0u>
{
public:
TkWallFunc() = default;
};

// =================================================================================================
// =================================================================================== TWallFuncBase

template<std::size_t d>
class TWallFuncBase
{
private:
TCampo<d, 0u> const &k;

// --------------------------------------------------------------------------------------- Funciones

protected:
TWallFuncBase(TCampo<d, 0u> const &k_) :
k(k_) {}

double
Uτ(TCara<d> const &Cara) const
{ return std::sqrt(std::sqrt(Cμ) * k.Eval(Cara.CeldaP())); }
};

// =================================================================================================
// ====================================================================================== TεWallFunc

template<std::size_t d, std::size_t r> requires (r == 0u)
class TεWallFunc : public TWallFuncBase<d>, public TCCBase<d, 0u>
{
private:
using TWallFuncBase<d>::Uτ;

public:
TεWallFunc() = delete;

TεWallFunc(TCampo<d, 0u> const &k_) :
TWallFuncBase<d>(k_) {}

std::tuple<double, TTensor<d, 0u>>
virtual Coef(TCara<d> const &) const override
{ return {}; }

std::tuple<double, TTensor<d, 0u>>
virtual GradCoef(TCara<d> const &Cara) const override
{ TVector const L = Cara.L(); return {0.0, pow<3u>(Uτ(Cara)) / (κ * (L & L))}; }
};

// =================================================================================================
// ==================================================================================== TμEfWallFunc

template<std::size_t d, std::size_t r> requires (r == 0u)
class TμEfWallFunc : public TWallFuncBase<d>, public TCCBase<d, 0u>
{
private:
double const μ;

// --------------------------------------------------------------------------------------- Funciones

private:
using TWallFuncBase<d>::Uτ;

public:
TμEfWallFunc() = delete;

TμEfWallFunc(TCampo<d, 0u> const &k_, double const μ_) :
TWallFuncBase<d>(k_), μ(μ_) {}

std::tuple<double, TTensor<d, 0u>>
virtual Coef(TCara<d> const &) const override;
};

// ======================================================================== IMPLEMENTACIÓN DE CLASES
// ==================================================================================== TμEfWallFunc

template<std::size_t d, std::size_t r> requires (r == 0u)
std::tuple<double, TTensor<d, 0u>>
TμEfWallFunc<d, r>::Coef(TCara<d> const &Cara) const
{
double const yPlus = Uτ(Cara) * mag(Cara.L()) / μ;

return {0.0, μ * κ * yPlus / std::log(E * yPlus)};
}

} // VF
