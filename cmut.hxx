#ifndef MUT_HXX
#define MUT_HXX

#pragma region Imports

#include <bit>
#include <random>
#include <type_traits>
#include <immintrin.h>
#include <cstdint>

#pragma endregion

#pragma region Preprocessor

#pragma runtime_checks("scu", off)		// Disable runtime checks
#pragma strict_gs_check(off)			// Disable stack protection
#pragma optimize("s", on)				// Force optimization
#pragma inline_depth(255)		  
#pragma inline_recursion(on)

#pragma endregion

#pragma region Macros

#define rshl(BASE, MODIFIER) static_cast<decltype(BASE)>( \
	((BASE << ((MODIFIER) % (sizeof(BASE) * 8))) | (BASE >> ((sizeof(BASE) * 8) - ((MODIFIER) % (sizeof(BASE) * 8))))))
#define rshr(BASE, MODIFIER) static_cast<decltype(BASE)>( \
	((BASE >> ((MODIFIER) % (sizeof(BASE) * 8))) | (BASE << ((sizeof(BASE) * 8) - ((MODIFIER) % (sizeof(BASE) * 8))))))

#pragma endregion

typedef enum mut_t : std::uint8_t {

	i8		= 0,
	ui8		= 1,

	i16		= 2,
	ui16	= 3,

	i32		= 4,
	ui32	= 5,

	i64		= 6,
	ui64	= 7,

	f32		= 8,
	f64		= 9,

	ui8_arr = 11,

	err_t	= 12
};

template<typename T>
static inline __forceinline constexpr const mut_t get_mut_t(const T object) noexcept {

	using base_t = std::remove_cv_t<T>;

	if		constexpr (std::is_same_v<base_t, std::int8_t>)
		return i8;
	else if constexpr (std::is_same_v<base_t, std::uint8_t>)
		return ui8;
	else if constexpr (std::is_same_v<base_t, std::int16_t>)
		return i16;
	else if constexpr (std::is_same_v<base_t, std::uint16_t>)
		return ui16;
	else if constexpr (std::is_same_v<base_t, std::int32_t>)
		return i32;
	else if constexpr (std::is_same_v<base_t, std::uint32_t>)
		return ui32;
	else if constexpr (std::is_same_v<base_t, std::int64_t>)
		return i64;
	else if constexpr (std::is_same_v<base_t, std::uint64_t>)
		return ui64;
	else if constexpr (std::is_same_v<base_t, float>)
		return f32;
	else if constexpr (std::is_same_v<base_t, double> || std::is_same_v<base_t, long double>)
		return f64;
	else if constexpr (std::is_same_v<base_t, bool>)
		return ui8;

	return err_t;
}

// Returns the Distance from MSB which Contained the Last True Bit
template<typename T>
static inline __forceinline const std::uint8_t __fastcall max_headroom_t(const T object) noexcept {

	std::uint8_t r_position = 0;
	for (std::size_t i = 0; i < sizeof(T) * 8; ++i)
		if (object & (T(0x1) << i))
			r_position = i;

	return r_position;
}

template<typename T>
class cmut {

private:

#pragma region Base Globals

	const mut_t								base_type;

	const std::uint32_t						seed = std::mt19937(std::time(nullptr))();

#pragma endregion

#pragma region Binary Mutation Trackers

	// How many bits were Shifted Left per each Mutated Form
	std::uint8_t							m_sh_16;
	std::uint8_t							m_sh_32;
	std::uint8_t							m_sh_64;
	std::uint8_t							m_sh_v128;
	std::uint8_t							m_sh_v128_64;

#pragma endregion

#pragma region Mutated / Polymorphic Storage

	// Bits are Stored in Reverse-Order Low-High
	bool									original_set_map[128]{ false };

	std::uint16_t							m_ui16;

	std::uint8_t							m_rsh_ui32;
	std::uint32_t							m_ui32;

	alignas(0x2) std::uint16_t				m_ui64_split16[4];
	alignas(0x4) std::uint32_t				m_ui64_split32[2];

	// m_v128 is used to Mutate Lower Primitive Types
	__m128i									m_v128;

	// Fallback to Bitmap for 64-bit or Circular-Rotate 2 32-bit Integers, m_rsh_ui64 Descripts amount of Bits Circularly Rotated per-Subword split Word (both Subword 32/16)
	std::uint8_t							m_rsh_ui64;
	std::uint64_t							m_ui64;

#pragma endregion

#pragma region Control Flow Flags / Modulators

	/*
		0 = SideWord / SubWord Mutation Reconstruction
		1 = Mapping Reconstruction
	*/
	const bool								reconstruct_mode	= static_cast<bool>(seed % 2);

	const std::uint8_t						rec_ui8_mode		= seed % 4;
	const std::uint8_t						rec_ui16_mode		= seed % 3;
	const std::uint8_t						rec_ui32_mode		= seed % 2;
	const std::uint8_t						rec_ui64_mode		= rec_ui16_mode;

#pragma endregion

#pragma region Type Deconstruction / Mutation(s)

	template<typename _T>
	inline __forceinline void deconstruct_t(volatile _T object) noexcept {

		std::mt19937		r(std::time(nullptr));

		const std::size_t	headroom		= max_headroom_t(object);

		const std::size_t	max_pos_ui16	= (sizeof(std::uint16_t) * 8) - 1 - headroom;
		const std::size_t	max_pos_ui32	= (sizeof(std::uint32_t) * 8) - 1 - headroom;
		const std::size_t	max_pos_ui64	= (sizeof(std::uint64_t) * 8) - 1 - headroom;

		m_sh_16								= r() % (max_pos_ui16 + 1);
		m_sh_32								= r() % (max_pos_ui32 + 1);
		m_sh_64								= r() % (max_pos_ui64 + 1);
		m_sh_v128							= r() % (sizeof(__m128i) / sizeof(std::uint32_t));

		std::uint8_t		dummy_bytect16	= m_sh_16 / 8;
		std::uint8_t		dummy_bytect32	= m_sh_32 / 8;
		std::uint8_t		dummy_bytect64	= m_sh_64 / 8;

		// Fill Unused Bytes with Randomized Data
		for (std::size_t i = 0; i < dummy_bytect16; ++i)
			m_ui16 |= static_cast<std::uint16_t>(static_cast<std::uint8_t>(r() & 0xFF)) << (i * 8);

		for (std::size_t i = 0; i < dummy_bytect32; ++i)
			m_ui16 |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(r() & 0xFF)) << (i * 8);

		for (std::size_t i = 0; i < dummy_bytect64; ++i)
			m_ui16 |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(r() & 0xFF)) << (i * 8);

		switch (base_type) {

			case i8:
			case ui8: {

				m_ui16					= static_cast<std::uint16_t>(object) << m_sh_16;
				m_ui32					= static_cast<std::uint32_t>(object) << m_sh_32;
				m_ui64					= static_cast<std::uint64_t>(object) << m_sh_64;

				m_v128 = _mm_set_epi32(

					m_sh_v128 == 3 ? static_cast<std::uint32_t>(object) << m_sh_32 : r(),
					m_sh_v128 == 2 ? static_cast<std::uint32_t>(object) << m_sh_32 : r(),
					m_sh_v128 == 1 ? static_cast<std::uint32_t>(object) << m_sh_32 : r(),
					m_sh_v128 == 0 ? static_cast<std::uint32_t>(object) << m_sh_32 : r()
				);


				break;
			}
			case i16:
			case ui16: {

				m_ui32					= static_cast<std::uint32_t>(object) << m_sh_32;
				m_ui64					= static_cast<std::uint64_t>(object) << m_sh_64;

				m_v128 = _mm_set_epi32(

					m_sh_v128 == 3 ? static_cast<std::uint32_t>(object) << m_sh_32 : r(),
					m_sh_v128 == 2 ? static_cast<std::uint32_t>(object) << m_sh_32 : r(),
					m_sh_v128 == 1 ? static_cast<std::uint32_t>(object) << m_sh_32 : r(),
					m_sh_v128 == 0 ? static_cast<std::uint32_t>(object) << m_sh_32 : r()
				);

				break;
			}
			case f32:
			case i32:
			case ui32: {

				m_rsh_ui32				= r();
				m_ui64					= static_cast<std::uint64_t>(object) << m_sh_64;

				m_v128 = _mm_set_epi32(

					m_sh_v128 == 3 ? rshl(object, m_rsh_ui32) : r(),
					m_sh_v128 == 2 ? rshl(object, m_rsh_ui32) : r(),
					m_sh_v128 == 1 ? rshl(object, m_rsh_ui32) : r(),
					m_sh_v128 == 0 ? rshl(object, m_rsh_ui32) : r()
				);

				break;
			}
			case f64:
			case i64:
			case ui64:{

				m_rsh_ui64				= r();
				m_sh_v128_64			= m_sh_v128 % 2;

				for (std::size_t i = 0; i < 4; ++i) {

					auto sw16			= reinterpret_cast<volatile std::uint16_t*>(&object)[i];
					m_ui64_split16[i]	= rshl(sw16, m_rsh_ui64);
				}

				for (std::size_t i = 0; i < 2; ++i) {

					auto sw32			= reinterpret_cast<volatile std::uint32_t*>(&object)[i];
					m_ui64_split32[i]	= rshl(sw32, m_rsh_ui64);
				}

				m_v128 = _mm_set_epi64x(
					
					m_sh_v128_64 == 0 ? rshl(object, m_rsh_ui64) : r() * r(),
					m_sh_v128_64 == 1 ? rshl(object, m_rsh_ui64) : r() * r()
				);

				break;
			}
			
			default: break;
		}
	}

#pragma endregion

#pragma region Type Reconstruction (Inverse-Mutation)

	template<typename _T>
	inline __forceinline const _T __fastcall reconstruct_t() noexcept {

		using r_T = std::remove_cv_t<_T>;

		volatile r_T r_T_inst = _T(NULL);

		if (reconstruct_mode) {

			for (std::size_t i = 0; i < (sizeof(T) * 8); ++i)
				r_T_inst	  |= static_cast<r_T>(original_set_map[i]) << i;
		}
		else {

			switch (base_type) {

			case i8:
			case ui8: {

				switch (rec_ui8_mode) {

					case 0: {

						r_T_inst = m_ui16 >> m_sh_16;
						break;
					}
					case 1: {

						r_T_inst = m_ui32 >> m_sh_32;
						break;
					}
					case 2: {

						r_T_inst = m_ui64 >> m_sh_64;
						break;
					}
					case 3: {

						alignas(0x10) std::uint32_t arr_v128[4];

						_mm_store_si128(reinterpret_cast<__m128i*>(arr_v128), m_v128);

						r_T_inst = static_cast<r_T>(arr_v128[m_sh_v128] >> m_sh_32);
						break;
					}
				default: break;
				}
				break;
			}
			case i16:
			case ui16: {

				switch (rec_ui16_mode) {

					case 0: {

						r_T_inst = m_ui32 >> m_sh_32;
						break;
					}
					case 1: {

						r_T_inst = m_ui64 >> m_sh_64;
						break;
					}
					case 2: {

						alignas(0x10) std::uint32_t arr_v128[4];

						_mm_store_si128(reinterpret_cast<__m128i*>(arr_v128), m_v128);

						r_T_inst = static_cast<r_T>(arr_v128[m_sh_v128] >> m_sh_32);
						break;
					}
				}

				break;
			}
			case f32:
			case i32:
			case ui32: {

				if (rec_ui32_mode) {

					r_T_inst = static_cast<r_T>(m_ui64 >> m_sh_64);
				}
				else {

					alignas(0x10) std::uint32_t arr_v128[4];

					_mm_store_si128(reinterpret_cast<__m128i*>(arr_v128), m_v128);

					r_T_inst = rshr(static_cast<r_T>(arr_v128[m_sh_v128]), m_rsh_ui32);
				}

				break;
			}
			case f64:
			case i64:
			case ui64: {

				switch (rec_ui64_mode) {

					case 0: {

						for (std::size_t i = 0; i < 2; ++i) {

							const auto sw32 = m_ui64_split32[i];
							reinterpret_cast<volatile std::uint32_t*>(&r_T_inst)[i] = rshr(sw32, m_rsh_ui64);
						}
						break;
					}
					case 1: {

						for (std::size_t i = 0; i < 4; ++i) {

							const auto sw16 = m_ui64_split16[i];
							reinterpret_cast<volatile std::uint16_t*>(&r_T_inst)[i] = rshr(sw16, m_rsh_ui64);
						}
						break;
					}
					case 2: {

						alignas(0x10) std::uint64_t arr_v128[2];
						_mm_store_si128(reinterpret_cast<__m128i*>(arr_v128), m_v128);

						r_T_inst = rshr(static_cast<std::remove_cv_t<_T>>(arr_v128[m_sh_v128_64 ? 0 : 1]), m_rsh_ui64);
						break;
					}
				}

				break;
			}

			default: break;
			}
		}

		return r_T_inst;
	}

#pragma endregion

public:

	inline __forceinline cmut(const T object) noexcept : base_type(get_mut_t(object)) {

		set(object);
	}

	inline __forceinline const std::remove_cv_t<T> get() noexcept {
		if constexpr (std::is_same_v<std::remove_cv_t<T>, float>)
			return std::bit_cast<float>(reconstruct_t<std::uint32_t>());
		else if constexpr (std::is_same_v<std::remove_cv_t<T>, double>)
			return std::bit_cast<double>(reconstruct_t<std::uint64_t>());
		else if constexpr (std::is_same_v<std::remove_cv_t<T>, long double>)
			return std::bit_cast<double>(reconstruct_t<std::uint64_t>());
		else
			return reconstruct_t<std::remove_cv_t<T>>();
	}

	inline __forceinline const bool set(const std::remove_cv_t<T> object) noexcept {

		if constexpr (!std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<decltype(object)>>)
			return false;

		std::mt19937 r(std::time(nullptr));

		if constexpr (std::is_same_v<std::remove_cv_t<T>, float>) {

			const volatile std::uint32_t v_obj = std::bit_cast<std::uint32_t>(object);

			for (std::size_t i = 0; i < (sizeof(T) * 8); ++i)
				original_set_map[i] = static_cast<bool>((v_obj >> i) & 0x1);

			deconstruct_t(v_obj);
		}
		else if constexpr (std::is_same_v<std::remove_cv_t<T>, double> || std::is_same_v<std::remove_cv_t<T>, long double>) {

			const volatile std::uint64_t v_obj = std::bit_cast<std::uint64_t>(object);

			for (std::size_t i = 0; i < (sizeof(T) * 8); ++i)
				original_set_map[i] = static_cast<bool>((v_obj >> i) & 0x1);

			deconstruct_t(v_obj);
		}
		else {

			const volatile T v_obj = *const_cast<T*>(&object);

			for (std::size_t i = 0; i < (sizeof(T) * 8); ++i)
				original_set_map[i] = static_cast<bool>((v_obj >> i) & 0x1);

			deconstruct_t(v_obj);
		}

		for (std::size_t i = (sizeof(T) * 8); i < sizeof(original_set_map); ++i)
			original_set_map[i] = static_cast<std::uint8_t>(r() & 0xFF);

		return true;
	}

	inline __forceinline ~cmut() noexcept {

		m_sh_16 = NULL;
		m_sh_32 = NULL;
		m_sh_64 = NULL;
		m_sh_v128 = NULL;
		m_sh_v128_64 = NULL;

		std::memset(original_set_map, NULL, sizeof(original_set_map));

		m_ui16 = NULL;

		m_ui32 = NULL;
		m_rsh_ui32 = NULL;

		m_ui64 = NULL;
		m_rsh_ui64 = NULL;
		std::memset(m_ui64_split16, NULL, sizeof(m_ui64_split16));
		std::memset(m_ui64_split32, NULL, sizeof(m_ui64_split32));

		m_v128 = _mm_setzero_si128();
	}

	inline __forceinline cmut __fastcall operator+(const std::remove_cv_t<T> value) const noexcept {
		return cmut(get() + value);
	}

	inline __forceinline cmut __fastcall operator-(const std::remove_cv_t<T> value) const noexcept {
		return cmut(get() - value);
	}

	inline __forceinline cmut __fastcall operator/(const std::remove_cv_t<T> value) const noexcept {
		return cmut(get() / value);
	}

	inline __forceinline cmut __fastcall operator*(const std::remove_cv_t<T> value) const noexcept {
		return cmut(get() * value);
	}

	inline __forceinline cmut __fastcall operator&(const std::remove_cv_t<T> value) const noexcept {
		return cmut(get() & value);
	}

	inline __forceinline cmut __fastcall operator|(const std::remove_cv_t<T> value) const noexcept {
		return cmut(get() | value);
	}

	inline __forceinline cmut __fastcall operator%(const std::remove_cv_t<T> value) const noexcept {
		return cmut(get() % value);
	}

	inline __forceinline cmut __fastcall operator^(const std::remove_cv_t<T> value) const noexcept {
		return cmut(get() ^ value);
	}

	inline __forceinline cmut __fastcall operator<<(const std::remove_cv_t<T> value) const noexcept {
		return cmut(get() << value);
	}

	inline __forceinline cmut __fastcall operator>>(const std::remove_cv_t<T> value) const noexcept {
		return cmut(get() >> value);
	}

	inline __forceinline cmut& __fastcall operator+=(const std::remove_cv_t<T> value) noexcept {
		set(get() + value);
		return *this;
	}

	inline __forceinline cmut& __fastcall operator-=(const std::remove_cv_t<T> value) noexcept {
		set(get() - value);
		return *this;
	}

	inline __forceinline cmut& __fastcall operator++() noexcept {
		return this->operator+=(1);
	}

	inline __forceinline cmut& __fastcall operator--() noexcept {
		return this->operator-=(1);
	}

	inline __forceinline cmut& __fastcall operator*=(const std::remove_cv_t<T> value) noexcept {
		set(get() * value);
		return *this;
	}

	inline __forceinline cmut& __fastcall operator/=(const std::remove_cv_t<T> value) noexcept {
		set(get() / value);
		return *this;
	}

	inline __forceinline cmut& __fastcall operator%=(const std::remove_cv_t<T> value) noexcept {
		set(get() % value);
		return *this;
	}

	inline __forceinline cmut& __fastcall operator^=(const std::remove_cv_t<T> value) noexcept {
		set(get() ^ value);
		return *this;
	}

	inline __forceinline cmut& __fastcall operator&=(const std::remove_cv_t<T> value) noexcept {
		set(get() & value);
		return *this;
	}

	inline __forceinline cmut& __fastcall operator|=(const std::remove_cv_t<T> value) noexcept {
		set(get() | value);
		return *this;
	}

	inline __forceinline cmut& __fastcall operator<<=(const std::remove_cv_t<T> value) noexcept {
		set(get() << value);
		return *this;
	}

	inline __forceinline cmut& __fastcall operator>>=(const std::remove_cv_t<T> value) noexcept {
		set(get() >> value);
		return *this;
	}

	inline __forceinline cmut& __fastcall operator=(const std::remove_cv_t<T> value) noexcept {
		set(value);
		return *this;
	}

	inline __forceinline __stdcall operator std::remove_cv_t<T>() noexcept {
		return get();
	}

};

#endif