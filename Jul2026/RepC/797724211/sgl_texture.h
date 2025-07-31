#pragma once
#include <stdint.h>

#include <map>
#include <string>

#include "sgl_math.h"
#include "sgl_enums.h"

namespace sgl {

/**
Pixel color convert/pack/unpack.
  Color convertion. Vec4 <=> RGBA8.
  @param color: A Vec4 color (r,g,b,a), map value range [0.0, 1.0] to [0, 255],
  out of bound values will be clamped to 0 or 1 before conversion.
**/
inline void convert_Vec4_color_to_RGBA_uint8(
  const Vec4 & color, uint8_t & R, uint8_t & G, uint8_t & B, uint8_t & A)
{
  R = uint8_t(clamp(0, int(color.r * 255.0), 255));
  G = uint8_t(clamp(0, int(color.g * 255.0), 255));
  B = uint8_t(clamp(0, int(color.b * 255.0), 255));
  A = uint8_t(clamp(0, int(color.a * 255.0), 255));
}
inline void convert_RGBA_uint8_to_Vec4_color(
  const uint8_t & R, const uint8_t & G, const uint8_t & B, const uint8_t & A, Vec4 & color)
{
  color.r = double(R) / 255.0;
  color.g = double(G) / 255.0;
  color.b = double(B) / 255.0;
  color.a = double(A) / 255.0;
}
inline void pack_RGBA8888_to_uint32(
  const uint8_t & R, const uint8_t & G, const uint8_t & B, const uint8_t & A,
  const PixelFormat & target_format, uint32_t & out_result)
{
  /*
  note that here we default to little endian,
  the order of all color components should be reversed when packing
  */
  if (target_format == PixelFormat_RGBA8888)
    out_result = ((A << 24) | (B << 16) | (G << 8) | R);
  else if (target_format == PixelFormat_BGRA8888)
    out_result = ((A << 24) | (R << 16) | (G << 8) | B);
  else
    printf("Cannot pack pixel. Invalid texture format.\n");
}
inline void unpack_uint32_to_RGBA8888(
  const uint32_t & packed_uint32, const PixelFormat & source_format,
  uint8_t & R, uint8_t & G, uint8_t & B, uint8_t & A)
{
  if (source_format == PixelFormat_RGBA8888) {
    R = uint8_t(packed_uint32 & 0x000000FF);
    G = uint8_t((packed_uint32 & 0x0000FF00) >> 8);
    B = uint8_t((packed_uint32 & 0x00FF0000) >> 16);
    A = uint8_t((packed_uint32 & 0xFF000000) >> 24);
  }
  else if (source_format == PixelFormat_BGRA8888) {
    B = uint8_t(packed_uint32 & 0x000000FF);
    G = uint8_t((packed_uint32 & 0x0000FF00) >> 8);
    R = uint8_t((packed_uint32 & 0x00FF0000) >> 16);
    A = uint8_t((packed_uint32 & 0xFF000000) >> 24);
  }
  else
    printf("Cannot unpack pixel. Invalid texture format.\n");
}
inline Vec4 unpack_uint32_to_Vec4_color(
  const uint32_t & packed_uint32, const PixelFormat & source_format)
{
  uint8_t R, G, B, A;

  if (source_format == PixelFormat_RGBA8888) {
    R = uint8_t(packed_uint32 & 0x000000FF);
    G = uint8_t((packed_uint32 & 0x0000FF00) >> 8);
    B = uint8_t((packed_uint32 & 0x00FF0000) >> 16);
    A = uint8_t((packed_uint32 & 0xFF000000) >> 24);
  }
  else if (source_format == PixelFormat_BGRA8888) {
    B = uint8_t(packed_uint32 & 0x000000FF);
    G = uint8_t((packed_uint32 & 0x0000FF00) >> 8);
    R = uint8_t((packed_uint32 & 0x00FF0000) >> 16);
    A = uint8_t((packed_uint32 & 0xFF000000) >> 24);
  }
  else
    printf("Cannot unpack pixel. Invalid texture format.\n");

  Vec4 color;
  convert_RGBA_uint8_to_Vec4_color(R, G, B, A, color);
  return color;
}

class Texture {
  friend Vec4 texture(const Texture *texobj, const Vec2 &uv);
 protected:
  /* width, height, bytes per pixel */
  int32_t w, h, bypp; 
  /* raw pixel data */
  void *pixels; 
  /* pixel format, must be one of the types enumerated in `PixelFormat`. */
  PixelFormat format; 
  /* texture sampling method (such as nearest, linear, ...) */
  TextureSampling sampling; 
  /* some special textures have their own usage (such as depth buffers..) */
  TextureUsage usage;
  /* store comments for debugging */
  std::string comment;

 public:
  /**
  Create an empty texture.
  @param w, h: Width and height of the texture (in pixels).
  @param texture_format: Format of the created texture.
  @param texture_sampling: Defines how to interpolate texture data.
  **/
  void create(int32_t w, int32_t h, PixelFormat format, TextureSampling sampling, TextureUsage usage);
  /**
  Load an image from disk.
  **/
  void load(const std::string &file, const PixelFormat& target_format = PixelFormat_BGRA8888, const TextureSampling& texture_sampling = TextureSampling_Nearest, bool flip_vertically_on_load = false);
  /**
  Clear a texture.
  **/
  void clear(const Vec4& clear_color);
  /**
  Destroy texture.
  **/
  void destroy();
  /**
  Texture sampling (nearest sampling).
  @param p: Normalized texture coordinate. Origin is located at the lower-left
  corner, with +x pointing to the left and +y pointing to the top. Out of bound
  values will be clamped to [0, 1] before texturing.
  **/
  Vec4 texture_RGBA8888_point(const Vec2 &p) const;
  Vec4 texture_BGRA8888_point(const Vec2 &p) const;
  Vec4 texture_float64_point(const Vec2 &p) const;
  Vec4 texture_float32_point(const Vec2 &p) const;
  Vec4 texture_xxxx8888_bilinear(const Vec2 &p) const;
  Vec4 texture_float64_bilinear(const Vec2 &p) const;
  Vec4 texture_float32_bilinear(const Vec2 &p) const;
  Vec4 texture_uint8_point(const Vec2 &p) const;
  Vec4 texture_uint8_bilinear(const Vec2 &p) const;

  /**
  set/get
  **/
  void            set_sampling_mode(TextureSampling sampling) { this->sampling = sampling; }
  TextureSampling get_sampling_mode() const { return this->sampling; }
  int32_t                 get_width() const { return this->w; }
  int32_t                get_height() const { return this->h; }
  IVec2                    get_size() const { return IVec2(this->w, this->h); }
  int32_t       get_bytes_per_pixel() const { return this->bypp; }
  void*              get_pixel_data() const { return this->pixels; }
  PixelFormat      get_pixel_format() const { return this->format; }
  TextureUsage    get_texture_usage() const { return this->usage; }
  void            set_texture_usage(TextureUsage usage) { this->usage = usage; }
  void                  set_comment(const std::string& comment) { this->comment = comment; }
  std::string           get_comment() const { return this->comment; }

  /**
  Convert texture to another format.
  **/
  Texture to_format(const PixelFormat& target_format) const;
  /**
  Save texture to disk.
  **/
  bool save_png(const std::string& path) const;
  /**
  Flip texture vertically.
  **/
  void flip_vertically();

 protected:
  /**
  Internal copy from another texture.
  **/
  void copy(const Texture &texture);

 public:
  Texture();
  virtual ~Texture();
  Texture(const Texture &texture);
  Texture &operator=(const Texture &texture);
};

/**
  Create an empty image.
  This is equivalent to Texture::create().
**/
Texture create_texture(int32_t w, int32_t h, PixelFormat format, TextureSampling sampling, TextureUsage usage);

/**
  Load an image from disk and return the loaded texture object.
  @param file: Image file path.
  @returns: The loaded image texture. If image loading failed, an empty texture
will be returned (pixels=NULL).
**/
Texture load_texture(const std::string &file, const PixelFormat& target_format = PixelFormat_BGRA8888, const TextureSampling& texture_sampling = TextureSampling_Nearest, bool flip_vertically_on_load = false);

/**
  Common interface for sampling a texture. Designed mainly for fragment shaders.
  @param texobj: The texture object to be sampled.
  @param uv: Normalized texture coordinate. Out of bound values will be clipped.
  @returns: The sampled texture data returned as Vec4 (and will always returns Vec4).
**/
Vec4 texture(const Texture *texobj, const Vec2 &uv);

/**
  Bilinear interpolation implementation adapted from
  https://gist.github.com/folkertdev/6b930c7a7856e36dcad0a72a03e66716

  Both uv.x and uv.y MUST be in the range [0, 1].
**/
inline void bilinear_interpolation_xxxx8888(
  uint32_t* data, const PixelFormat& src_format,
  const int32_t input_width, const int32_t input_height,
  const Vec2& uv, Vec4* output)
{
  /* NOTE: uv must be in [0,1] */
  double qx = uv.x * input_width - 0.5;
  double qy = uv.y * input_height - 0.5;

  int32_t x_l = int32_t(qx);
  int32_t x_h = x_l + 1;
  int32_t y_l = int32_t(qy);
  int32_t y_h = y_l + 1;

  x_l = clamp(0, x_l, input_width - 1);
  x_h = clamp(0, x_h, input_width - 1);
  y_l = clamp(0, y_l, input_height - 1);
  y_h = clamp(0, y_h, input_height - 1);

  double dummy;
  double x_weight = modf(qx, &dummy);
  double y_weight = modf(qy, &dummy);

  Vec4 a = unpack_uint32_to_Vec4_color(data[(int)y_l * input_width + (int)x_l], src_format);
  Vec4 b = unpack_uint32_to_Vec4_color(data[(int)y_l * input_width + (int)x_h], src_format);
  Vec4 c = unpack_uint32_to_Vec4_color(data[(int)y_h * input_width + (int)x_l], src_format);
  Vec4 d = unpack_uint32_to_Vec4_color(data[(int)y_h * input_width + (int)x_h], src_format);

  Vec4 interped = \
    a * (1.0 - x_weight) * (1.0 - y_weight) +
    b * x_weight * (1.0 - y_weight) +
    c * y_weight * (1.0 - x_weight) +
    d * x_weight * y_weight;

  *output = interped;
}

template <typename T>
inline void bilinear_interpolation_scalar(T* data, 
  const int32_t input_width, const int32_t input_height,
  const Vec2& uv, Vec4* output)
{
  /* NOTE: uv must be in [0,1] */
  double qx = uv.x * input_width - 0.5;
  double qy = uv.y * input_height - 0.5;

  int32_t x_l = int32_t(qx);
  int32_t x_h = x_l + 1;
  int32_t y_l = int32_t(qy);
  int32_t y_h = y_l + 1;

  x_l = clamp(0, x_l, input_width - 1);
  x_h = clamp(0, x_h, input_width - 1);
  y_l = clamp(0, y_l, input_height - 1);
  y_h = clamp(0, y_h, input_height - 1);

  double dummy;
  double x_weight = modf(qx, &dummy);
  double y_weight = modf(qy, &dummy);

  T a = data[(int)y_l * input_width + (int)x_l];
  T b = data[(int)y_l * input_width + (int)x_h];
  T c = data[(int)y_h * input_width + (int)x_l];
  T d = data[(int)y_h * input_width + (int)x_h];

  double interped = \
    (double)a * (1.0 - x_weight) * (1.0 - y_weight) +
    (double)b * x_weight * (1.0 - y_weight) +
    (double)c * y_weight * (1.0 - x_weight) +
    (double)d * x_weight * y_weight;

  *output = Vec4(interped, interped, interped, 1.0);
}

/**
Texture Blitting refers to the process of transferring or copying a
portion of one texture (often an image or a surface) onto another.

Blitting involves a direct copy of pixel data from a source texture
to a destination texture or screen. The term "blit" is short for
"bit-block transfer", which originated from older graphics hardware
operations where pixels or blocks of memory were moved to other
areas without needing complex operations.

A more advanced feature related to blitting is sprite rendering,
which is implemented in `sgl_pass.h`. Sprite rendering extends
basic blitting by supporting additional operations such as scaling
and rotation.
**/
void blit_texture(sgl::Texture* source, sgl::Texture* target,
  int src_x, int src_y, int src_w, int src_h, int dst_x, int dst_y,
  sgl::Texture* src_mask = NULL, sgl::Texture* dst_mask = NULL);
void blit_texture(sgl::Texture* source, sgl::Texture* target,
  int src_x, int src_y, int src_w, int src_h, 
  int dst_x, int dst_y, int dst_w, int dst_h,
  sgl::Texture* src_mask = NULL, sgl::Texture* dst_mask = NULL);

/**
Fast texture scaling.

This function performs fast texture scaling, ignoring the texture's 
interpolation method setting and always using nearest sampling. It
returns the resized texture. The resized texture will retain the same 
sampling mode, usage, and format settings as the source texture.
**/
sgl::Texture resize_texture(sgl::Texture* source, 
  double scale_x, double scale_y);

}; /* namespace sgl */
