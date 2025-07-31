#pragma once

/**
 * @file sdf.hpp
 * @author karurochari
 * @brief Main repository for SDF-related functionality
 * @date 2025-03-10
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifdef SDF_USE_USM
    #pragma omp requires unified_shared_memory
#endif

//TODO: Remove the experimental feature by using a custom implementation
#include <experimental/type_traits>
#include <cstdlib>
#include <omp.h>

#include "utils/static.hpp"
#include "commons.hpp"

#define SDF_INTERNALS


template <class T, template <class...> class Template>
struct is_specialization : std::false_type {};

template <template <class...> class Template, class... Args>
struct is_specialization<Template<Args...>, Template> : std::true_type {};



namespace sdf{
    #ifdef SDF_DEFAULT_ATTRS
    typedef SDF_DEFAULT_ATTRS default_attrs;
    #else
    typedef idx_attrs<true> default_attrs;
    #endif

    
    //TODO complete helpers here
    template<field_t::type_t TYPE >
    std::pair<size_t,size_t> set_field(sdf_i auto& node, const path_t* paths, const char* name, const void * src){
        auto fields = paths==nullptr?node.fields():node.fields(paths);
        for(auto& field: fields){
            if(strcmp(name,field.name)==0 && field.type == TYPE){
                //SET & return good value
            }
        }
        return {0,0};
    }

    template<field_t::type_t TYPE >
    bool get_field(sdf_i auto& node, const path_t* paths, const char* name, void * dst){
        auto fields = paths==nullptr?node.fields():node.fields(paths);

        return false;
    }

    namespace utils{

        /**
         * @brief Alternative to std::shared_ptr to point to elements of a flat tree
         */
        //TODO: force alignment/padding to ensure layout is not affected in strange ways
        template<typename T>
        struct tree_idx_ref{
            uint16_t offset = 0;
            using element_type = T;
        };

        template <typename Attrs>
        struct tree_idx{
            using attrs_t = Attrs;

            inline Attrs operator()(const glm::vec3& pos) const;
            inline float sample(const glm::vec3& pos) const;

            inline void traits(traits_t&) const;
            inline const char* name() const;
            inline fields_t fields() const;
            inline fields_t fields(const path_t* steps) const;
            inline visibility_t is_visible() const;

            inline bool tree_visit_pre(const visitor_t& op);
            inline bool tree_visit_post(const visitor_t& op);
            inline bool ctree_visit_pre(const cvisitor_t& op) const;
            inline bool ctree_visit_post(const cvisitor_t& op) const;

            inline void* addr();
            inline const void* addr() const;
            inline size_t children() const;
        };

        struct empty_t{};

        template <typename Attrs, template<typename, typename... Args> typename T, typename... Args> requires sdf_i<T<Attrs,Args...>>
        using primitive = T<Attrs, Args...>;

        template <typename Attrs>
        struct base_dyn{
            using attrs_t = Attrs;
            virtual constexpr inline Attrs operator()(const glm::vec3& pos) const =0;
            virtual constexpr inline float sample(const glm::vec3& pos) const  =0;

            virtual constexpr inline void traits(traits_t&) const=0;
            virtual constexpr inline const char* name() const=0;
            virtual constexpr inline fields_t fields() const=0;
            virtual constexpr inline fields_t fields(const path_t* steps) const=0;
            virtual constexpr inline visibility_t is_visible() const=0;

            virtual constexpr bool tree_visit_pre(const visitor_t& op) = 0;
            virtual constexpr bool tree_visit_post(const visitor_t& op) = 0;
            virtual constexpr bool ctree_visit_pre(const cvisitor_t& op) const = 0;
            virtual constexpr bool ctree_visit_post(const cvisitor_t& op) const = 0;

            virtual constexpr void* addr()=0;
            virtual constexpr const void* addr() const=0;
            virtual constexpr size_t children() const=0;

            virtual uint64_t to_tree(tree::builder& dst)const=0;

            virtual ~base_dyn(){}
        };  

        template <typename Attrs, template<typename, typename... Args> typename T, typename... Args> requires sdf_i<T<Attrs,Args...>>
        struct dyn : T<Attrs, Args...>, base_dyn<Attrs>{
            dyn(const T<Attrs, Args...>& ref):T<Attrs,Args...>(ref){}

            virtual constexpr inline Attrs operator()(const glm::vec3& pos) const override{return static_cast<const T<Attrs, Args...>*>(this)->operator()(pos);}
            virtual constexpr inline float sample(const glm::vec3& pos) const override{return static_cast<const T<Attrs, Args...>*>(this)->sample(pos);}

            virtual constexpr inline void traits(traits_t& t) const override{return static_cast<const T<Attrs, Args...>*>(this)->traits(t);}
            virtual constexpr inline const char* name() const override{return static_cast<const T<Attrs, Args...>*>(this)->name();}
            virtual constexpr inline fields_t fields() const override{return static_cast<const T<Attrs, Args...>*>(this)->fields();}
            virtual constexpr inline fields_t fields(const path_t* steps) const override{return static_cast<const T<Attrs, Args...>*>(this)->fields(steps);}
            virtual constexpr inline visibility_t is_visible() const override{return static_cast<const T<Attrs, Args...>*>(this)->is_visible();}

            virtual constexpr inline bool tree_visit_pre(const visitor_t& op)override{return static_cast<T<Attrs, Args...>*>(this)->tree_visit_pre(op);};
            virtual constexpr inline bool tree_visit_post(const visitor_t& op)override{return static_cast<T<Attrs, Args...>*>(this)->tree_visit_post(op);};
            virtual constexpr inline bool ctree_visit_pre(const cvisitor_t& op)const override{return static_cast<const T<Attrs, Args...>*>(this)->ctree_visit_pre(op);};
            virtual constexpr inline bool ctree_visit_post(const cvisitor_t& op)const override{return static_cast<const T<Attrs, Args...>*>(this)->ctree_visit_post(op);};

            virtual constexpr inline void* addr()override{return static_cast<T<Attrs, Args...>*>(this)->addr();}
            virtual constexpr inline const void* addr()const override{return static_cast<const T<Attrs, Args...>*>(this)->addr();}
            virtual constexpr inline size_t children() const override{return static_cast<const T<Attrs, Args...>*>(this)->children();}

            virtual uint64_t to_tree(tree::builder& dst)const override{return static_cast<const T<Attrs, Args...>*>(this)->to_tree(dst);};
        };

        template <typename Attrs, typename T> requires sdf_i<T> 
        struct dyn_op : T, base_dyn<Attrs>{
            virtual constexpr inline Attrs operator()(const glm::vec3& pos) const override{return static_cast<const T*>(this)->operator()(pos);}
            virtual constexpr inline float sample(const glm::vec3& pos) const override{return static_cast<const T*>(this)->sample(pos);}

            virtual constexpr inline void traits(traits_t& t) const override{return static_cast<const T*>(this)->traits(t);}
            virtual constexpr inline void traits(const traits_t& l, const traits_t& r, traits_t& t) const {return static_cast<const T*>(this)->traits(l,r,t);}
            virtual constexpr inline const char* name() const override{return static_cast<const T*>(this)->name();}
            virtual constexpr inline fields_t fields() const override{return static_cast<const T*>(this)->fields();}
            virtual constexpr inline fields_t fields(const path_t* steps) const override{return static_cast<const T*>(this)->fields(steps);}
            virtual constexpr inline visibility_t is_visible() const override{return static_cast<const T*>(this)->is_visible();}

            virtual constexpr inline bool tree_visit_pre(const visitor_t& op)override{return  static_cast<T*>(this)->tree_visit_pre(op);};
            virtual constexpr inline bool tree_visit_post(const visitor_t& op)override{return  static_cast<T*>(this)->tree_visit_post(op);};
            virtual constexpr inline bool ctree_visit_pre(const cvisitor_t& op)const override{return  static_cast<const T*>(this)->ctree_visit_pre(op);};
            virtual constexpr inline bool ctree_visit_post(const cvisitor_t& op)const override{return  static_cast<const T*>(this)->ctree_visit_post(op);};

            virtual constexpr inline void* addr()override{return static_cast<T*>(this)->addr();}
            virtual constexpr inline const void* addr() const override{return static_cast<const T*>(this)->addr();}
            virtual constexpr inline size_t children() const override{return static_cast<const T*>(this)->children();}

            virtual constexpr uint64_t to_tree(tree::builder& dst)const override{return static_cast<const T*>(this)->to_tree(dst);};
            
            using T::T;
            using operation = T;
        };
      

        template <typename L, typename CFG = empty_t>
        struct unary_op{
                [[no_unique_address]] CFG cfg;

            protected:
                L _left;
                template <typename A>
                struct attrs{
                    using type = A::attrs_t;
                };

                template <typename A>
                struct attrs<std::shared_ptr<A>>{
                    using type = A::attrs_t;
                };

                template <typename A>
                struct attrs<tree_idx_ref<A>>{
                    using type = A::attrs_t;
                };

                template <typename A>
                struct attrs<tree_idx_ref<std::shared_ptr<A>>>{
                    using type = A::attrs_t;
                };

                //This block could probably be made prettier and without using experimental features.
                template<class C>
                using element_type_t = typename C::element_type;
                template<class S>
                using real_type = std::experimental::detected_or_t<S, element_type_t, S>;
                

            public:
                using cfg_t = CFG;
                using attrs_t = attrs<L>::type;

                inline constexpr unary_op(L left, const CFG& cfg):unary_op(left){
                    static_assert(!std::is_same<CFG, empty_t>(), "Cannot configure operators without configuration struct specified");
                    this->cfg=cfg;
                }

                inline constexpr unary_op(L left):_left(left){
                    if constexpr(is_specialization<L,std::shared_ptr>{} ){
                        static_assert(sdf_i<typename L::element_type>);
                    }
                    else if constexpr(is_specialization<L,tree_idx_ref>{}){
                        //static_assert(sdf_i<typename L::element_type>);
                    }
                    else{
                        static_assert(sdf_i<L>);
                    }
                }

                using LL = real_type<L>;

                constexpr inline const LL& left() const{
                    if constexpr(is_specialization<L,std::shared_ptr>{}) return *_left; 
                    else if constexpr(is_specialization<L,tree_idx_ref>{}) return *(LL*)((uint8_t*)(this)-_left.offset);  
                    else return _left;
                }

                constexpr inline LL& left(){
                    if constexpr(is_specialization<L,std::shared_ptr>{}) return *_left; 
                    else if constexpr(is_specialization<L,tree_idx_ref>{}) return *(LL*)((uint8_t*)(this)-_left.offset);  
                    else return _left;
                }
        };   


        template <typename L, typename R, typename CFG = empty_t>
        struct binary_op{
                [[no_unique_address]] CFG cfg;

            protected:
                L _left;
                R _right;

                template <typename A>
                struct attrs{
                    using type = A::attrs_t;
                };

                template <typename A>
                struct attrs<std::shared_ptr<A>>{
                    using type = A::attrs_t;
                };

                template <typename A>
                struct attrs<tree_idx_ref<A>>{
                    using type = A::attrs_t;
                };

                template <typename A>
                struct attrs<tree_idx_ref<std::shared_ptr<A>>>{
                    using type = A::attrs_t;
                };

                //This block could probably be made prettier and without using experimental features.
                template<class C>
                using element_type_t = typename C::element_type;
                template<class S>
                using real_type = std::experimental::detected_or_t<S, element_type_t, S>;

            public:
                using cfg_t = CFG;
                using attrs_t = attrs<L>::type;

                inline constexpr binary_op(L left, R right):_left(left),_right(right){
                    if constexpr(is_specialization<L,std::shared_ptr>{}){
                        static_assert(std::is_same<typename L::element_type::attrs_t,typename R::element_type::attrs_t>() );
                        static_assert(sdf_i<typename L::element_type>);
                        static_assert(sdf_i<typename R::element_type>);
                    }
                    else if constexpr(is_specialization<L,tree_idx_ref>{}){
                        //TODO: Find a decent way sto make these static assertions work as intended
                        //static_assert(std::is_same<typename L::element_type::attrs_t,typename R::element_type::attrs_t>() );
                        //static_assert(sdf_i<typename L::element_type>);
                        //static_assert(sdf_i<typename R::element_type>);
                    }
                    else{
                        static_assert(std::is_same<typename L::attrs_t,typename R::attrs_t>() );
                        static_assert(sdf_i<L>);
                        static_assert(sdf_i<R>);
                    }
                }

                inline constexpr binary_op(L left, R right, const CFG& cfg):binary_op(left,right){
                    static_assert(!std::is_same<CFG, empty_t>(), "Cannot configure operators without configuration struct specified");
                    this->cfg=cfg;
                }


                using LL = real_type<L>;
                using RR = real_type<R>;

                constexpr inline const LL& left() const{
                    if constexpr(is_specialization<L,std::shared_ptr>{}) return *_left;
                    else if constexpr(is_specialization<L,tree_idx_ref>{}) return *(LL*)((uint8_t*)(this)-_left.offset);  
                    else return _left;
                }
                constexpr inline const RR& right() const{
                    if constexpr(is_specialization<R,std::shared_ptr>{}) return *_right;
                    else if constexpr(is_specialization<R,tree_idx_ref>{}) return *(RR*)((uint8_t*)(this)-_right.offset);  
                    else return _right;
                }

                constexpr inline LL& left(){
                    if constexpr(is_specialization<L,std::shared_ptr>{}) return *_left;
                    else if constexpr(is_specialization<L,tree_idx_ref>{}) return *(LL*)((uint8_t*)(this)-_left.offset);  
                    else return _left;
                }
                constexpr inline RR& right(){
                    if constexpr(is_specialization<R,std::shared_ptr>{}) return *_right;
                    else if constexpr(is_specialization<R,tree_idx_ref>{}) return *(RR*)((uint8_t*)(this)-_right.offset);  
                    else return _right;
                }

        };   


   
    }

    namespace{
    namespace impl_base{
        using namespace glm;
    }}

    namespace{namespace impl{
        using namespace glm;
    }}


    //Dynamic is not needed in base mode, as smart pointers on which it is built upon are not C.

    /**
     * @brief Static implementation, which can be resolved at compile time
     * 
     */
    namespace comptime{}

    /**
     * @brief Polymorphic implementation, moslty used as a bridge for dynamic
     * 
     */
    namespace polymorphic{}

    /**
     * @brief Dynamic implementation, its provides SDF tree which an be edited at runtime.
     * 
     */
    namespace dynamic{}

}


/* #region macro-machinery */

/// Used to record a primitive `NAME` across all namespaces
#define sdf_register_primitive(NAME)                                                                            \
namespace{                                                                                                      \
    namespace impl{                                                                                             \
    template <typename Attrs=default_attrs>                                                                     \
    struct NAME : impl_base::NAME<Attrs>{                                                                       \
        using attrs_t = Attrs;                                                                                  \
        uint64_t to_tree(tree::builder& dst)const;                                                              \
        using impl_base::NAME<Attrs>::fields;                                                                   \
        inline fields_t fields(const path_t* steps) const;                                                      \
        constexpr bool tree_visit_pre(const visitor_t& op);                                                     \
        constexpr bool tree_visit_post(const visitor_t& op);                                                    \
        constexpr bool ctree_visit_pre(const cvisitor_t& op) const;                                             \
        constexpr bool ctree_visit_post(const cvisitor_t& op) const;                                            \
        constexpr inline void* addr(){return (void*)this;}                                                      \
        constexpr inline const void* addr()const{return (const void*)this;}                                     \
        constexpr inline size_t children() const{return 0;}                                                     \
        using impl_base::NAME<Attrs>::NAME;                                                                     \
    };                                                                                                          \
    template <typename Attrs>                                                                                   \
    fields_t  NAME <Attrs> :: fields(const path_t* steps)const {                                                \
        if(steps[0]==END){                                                                                      \
            return this->fields();                                                                              \
        }                                                                                                       \
        return {nullptr,0};                                                                                     \
    }                                                                                                           \
    template <typename Attrs>                                                                                   \
    constexpr bool NAME <Attrs> :: tree_visit_pre(const visitor_t& op){                                         \
        return op(this->name(),this->fields(),this->addr(),this->children());                                   \
    }                                                                                                           \
    template <typename Attrs>                                                                                   \
    constexpr bool NAME <Attrs> :: tree_visit_post(const visitor_t& op){                                        \
        return op(this->name(),this->fields(),this->addr(),this->children());                                   \
    }                                                                                                           \
    template <typename Attrs>                                                                                   \
    constexpr bool NAME <Attrs> :: ctree_visit_pre(const cvisitor_t& op) const{                                 \
        return op(this->name(),this->fields(),this->addr(),this->children());                                   \
    }                                                                                                           \
    template <typename Attrs>                                                                                   \
    constexpr bool NAME <Attrs> :: ctree_visit_post(const cvisitor_t& op) const{                                \
        return op(this->name(),this->fields(),this->addr(),this->children());                                   \
    }                                                                                                           \
    template <typename Attrs>                                                                                   \
    uint64_t  NAME <Attrs> :: to_tree(tree::builder& dst)const {                                                \
        auto idx= dst.push(tree::op_t:: NAME, (uint8_t*)this, sizeof( NAME<Attrs> ));                           \
        return idx;                                                                                             \
    }                                                                                                           \
}                                                                                                               \
}                                                                                                               \
namespace comptime {                                                                                            \
    template <typename Attrs=default_attrs>                                                                     \
    using NAME##_t = utils::primitive<Attrs,impl::NAME >;                                                       \
    template <typename Attrs=default_attrs>                                                                     \
    constexpr inline NAME##_t<Attrs> NAME (  impl::NAME<Attrs> && ref ){                                        \
        return ref;                                                                                             \
    }                                                                                                           \
}                                                                                                               \
namespace polymorphic {                                                                                         \
    template <typename Attrs=default_attrs>                                                                     \
    using NAME##_t = utils::dyn<Attrs,impl::NAME >;                                                             \
    template <typename Attrs=default_attrs>                                                                     \
    constexpr inline NAME##_t<Attrs> NAME (  impl::NAME<Attrs> && ref ){                                        \
        return ref;                                                                                             \
    }                                                                                                           \
}                                                                                                               \
namespace dynamic {                                                                                             \
    template <typename Attrs=default_attrs>                                                                     \
    using NAME##_t =utils::dyn<Attrs,impl::NAME >;                                                              \
    template <typename Attrs=default_attrs>                                                                     \
    constexpr inline std::shared_ptr<utils::base_dyn<Attrs>> NAME (  impl::NAME<Attrs> && ref ){                \
        std::shared_ptr<utils::base_dyn<Attrs>> tmp =                                                           \
            std::make_shared<utils::dyn<Attrs,impl::NAME>>(utils::dyn<Attrs,impl::NAME>(ref));                  \
        return tmp;                                                                                             \
    }                                                                                                           \
}                                                                                                               \


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//TODO: Direct dependencies from configs::XXX have been removed in operator_1 to allow arbitrary (template) cfg. 
//      Adapt here as well first time this is needed for a binary operator.

/// Used to record a binary operator `NAME` across all namespaces

#define sdf_register_operator_2(NAME)                                                                           \
namespace{                                                                                                      \
namespace impl{                                                                                                 \
    template <typename A, typename B>                                                                           \
    struct NAME : impl_base::NAME<A,B>{                                                                         \
        using typename impl_base::NAME<A,B>::attrs_t;                                                           \
        uint64_t to_tree(tree::builder& dst)const;                                                              \
        using impl_base::NAME<A,B>::fields;                                                                     \
        inline fields_t fields(const path_t* steps) const;                                                      \
        constexpr bool tree_visit_pre(const visitor_t& op);                                                     \
        constexpr bool tree_visit_post(const visitor_t& op);                                                    \
        constexpr bool ctree_visit_pre(const cvisitor_t& op)const;                                              \
        constexpr bool ctree_visit_post(const cvisitor_t& op)const;                                             \
        constexpr inline void* addr(){return (void*)this;}                                                      \
        constexpr inline const void* addr()const{return (const void*)this;}                                     \
        constexpr inline size_t children() const{return 2;}                                                     \
        using impl_base::NAME<A,B>::NAME;                                                                       \
        using typename impl_base::NAME<A,B>::base;                                                              \
    };                                                                                                          \
    template<typename A, typename B>                                                                            \
    fields_t NAME <A,B> :: fields(const path_t* steps)const {                                                   \
        if(steps[0]==END){                                                                                      \
            return this->fields();                                                                              \
        }                                                                                                       \
        else{                                                                                                   \
            if(steps[0]==LEFT){return this->left().fields(steps+1);}                                            \
            else {return this->right().fields(steps+1);}                                                        \
        }                                                                                                       \
        return {nullptr,0};                                                                                     \
    }                                                                                                           \
    template<typename A, typename B>                                                                            \
    constexpr bool NAME <A,B> :: tree_visit_pre(const visitor_t& op){                                           \
        auto sval = op(this->name(),this->fields(), this->addr(), this->children());                            \
        auto lval = this->left().tree_visit_pre(op);                                                            \
        auto rval = this->right().tree_visit_pre(op);                                                           \
        return sval && lval && rval;                                                                            \
    }                                                                                                           \
    template<typename A, typename B>                                                                            \
    constexpr bool NAME <A,B> :: tree_visit_post(const visitor_t& op){                                          \
        auto lval = this->left().tree_visit_post(op);                                                           \
        auto rval = this->right().tree_visit_post(op);                                                          \
        auto sval = op(this->name(),this->fields(), this->addr(), this->children());                            \
        return sval && lval && rval;                                                                            \
    }                                                                                                           \
    template<typename A, typename B>                                                                            \
    constexpr bool NAME <A,B> :: ctree_visit_pre(const cvisitor_t& op)const{                                    \
        auto sval = op(this->name(),this->fields(), this->addr(), this->children());                            \
        auto lval = this->left().ctree_visit_pre(op);                                                           \
        auto rval = this->right().ctree_visit_pre(op);                                                          \
        return sval && lval && rval;                                                                            \
    }                                                                                                           \
    template<typename A, typename B>                                                                            \
    constexpr bool NAME <A,B> :: ctree_visit_post(const cvisitor_t& op)const{                                   \
        auto lval = this->left().ctree_visit_post(op);                                                          \
        auto rval = this->right().ctree_visit_post(op);                                                         \
        auto sval = op(this->name(),this->fields(), this->addr(), this->children());                            \
        return sval && lval && rval;                                                                            \
    }                                                                                                           \
    template<typename A, typename B>                                                                            \
    uint64_t NAME <A,B> :: to_tree(tree::builder& dst)const {                                                   \
        auto lname= base::left().to_tree(dst);                                                                  \
        auto rname = base::right().to_tree(dst);                                                                \
        if constexpr(std::is_same<typename base::cfg_t, utils::empty_t>()){                                     \
            NAME <utils::tree_idx_ref<utils::tree_idx<A>>,utils::tree_idx_ref<utils::tree_idx<B>>> tmp({(uint16_t)(dst.next()-lname)},{(uint16_t)(dst.next()-rname)});          \
            auto ret = dst.push(tree::op_t:: NAME, (uint8_t*)&tmp, sizeof(decltype(tmp)));                      \
            return ret;                                                                                         \
        }                                                                                                       \
        else{                                                                                                   \
            NAME <utils::tree_idx_ref<utils::tree_idx<A>>,utils::tree_idx_ref<utils::tree_idx<B>>> tmp({(uint16_t)(dst.next()-lname)},{(uint16_t)(dst.next()-rname)}, this->cfg);\
            auto ret = dst.push(tree::op_t:: NAME, (uint8_t*)&tmp, sizeof(decltype(tmp)));                      \
            return ret;                                                                                         \
        }                                                                                                       \
    }                                                                                                           \
}}                                                                                                              \
namespace comptime {                                                                                            \
    constexpr inline auto NAME(auto&& a, auto&& b){                                                             \
        using left_t = std::remove_reference<decltype(a)>::type;                                                \
        using right_t = std::remove_reference<decltype(b)>::type;                                               \
        static_assert(std::derived_from<impl:: NAME <left_t,right_t>, utils::binary_op<left_t,right_t>>, "Operator must be derived from binary_op");\
        return impl::NAME <left_t, right_t>(a,b);                                                               \
    };                                                                                                          \
    constexpr inline auto NAME(auto&& a, auto&& b, const typename configs:: NAME& cfg){                         \
        using left_t = std::remove_reference<decltype(a)>::type;                                                \
        using right_t = std::remove_reference<decltype(b)>::type;                                               \
        static_assert(std::derived_from<impl:: NAME <left_t,right_t>, utils::binary_op<left_t,right_t,configs:: NAME>>, "Operator must be derived from binary_op");\
        return impl::NAME <left_t, right_t>(a,b,cfg);                                                           \
    };                                                                                                          \
}                                                                                                               \
namespace polymorphic {                                                                                         \
    template<typename Attrs=default_attrs>                                                                      \
    constexpr inline auto NAME(auto&& a, auto&& b){                                                             \
        using left_t = std::remove_reference<decltype(a)>::type;                                                \
        using right_t = std::remove_reference<decltype(b)>::type;                                               \
        return utils::dyn_op<Attrs,impl::NAME<left_t,right_t>>(a,b);                                            \
    };                                                                                                          \
    template<typename Attrs=default_attrs>                                                                      \
    constexpr inline auto NAME(auto&& a, auto&& b, const typename configs:: NAME& cfg){                         \
        using left_t = std::remove_reference<decltype(a)>::type;                                                \
        using right_t = std::remove_reference<decltype(b)>::type;                                               \
        static_assert(std::derived_from<impl:: NAME <left_t,right_t>, utils::binary_op<left_t,right_t,configs:: NAME>>, "Operator must be derived from binary_op");\
        return utils::dyn_op<Attrs,impl::NAME <left_t, right_t>>(a,b,cfg);                                      \
    };                                                                                                          \
}                                                                                                               \
namespace dynamic {                                                                                             \
    template<typename Attrs=default_attrs>                                                                      \
    using NAME##_t=utils::dyn_op<Attrs,impl::NAME<std::shared_ptr<utils::base_dyn<Attrs>>,std::shared_ptr<utils::base_dyn<Attrs>>>>;\
    template<typename Attrs=default_attrs>                                                                      \
    constexpr inline std::shared_ptr<utils::base_dyn<Attrs>> NAME                                               \
    (                                                                                                           \
        std::shared_ptr<utils::base_dyn<Attrs>> a,                                                              \
        std::shared_ptr<utils::base_dyn<Attrs>> b                                                               \
    ){                                                                                                          \
        return std::make_shared<utils::dyn_op<Attrs,impl::NAME<decltype(a),decltype(b)>>>(a,b);                 \
    }                                                                                                           \
    template<typename Attrs=default_attrs>                                                                      \
    constexpr inline std::shared_ptr<utils::base_dyn<Attrs>> NAME                                               \
    (                                                                                                           \
        std::shared_ptr<utils::base_dyn<Attrs>> a,                                                              \
        std::shared_ptr<utils::base_dyn<Attrs>> b,                                                              \
        const typename configs:: NAME& cfg                                                                      \
    ){                                                                                                          \
        return std::make_shared<utils::dyn_op<Attrs,impl::NAME<decltype(a),decltype(b)>>>(a,b,cfg);             \
    }                                                                                                           \
}                                                                                                               \


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#define sdf_register_operator_1(NAME)                                                                           \
namespace{                                                                                                      \
namespace impl{                                                                                                 \
    template <typename A>                                                                                       \
    struct NAME : impl_base::NAME<A>{                                                                           \
        using typename impl_base::NAME<A>::attrs_t;                                                             \
        uint64_t to_tree(tree::builder& dst)const;                                                              \
        using impl_base::NAME<A>::fields;                                                                       \
        inline fields_t fields(const path_t* steps) const;                                                      \
        constexpr bool tree_visit_pre(const visitor_t& op);                                                     \
        constexpr bool tree_visit_post(const visitor_t& op);                                                    \
        constexpr bool ctree_visit_pre(const cvisitor_t& op)const;                                              \
        constexpr bool ctree_visit_post(const cvisitor_t& op)const;                                             \
        constexpr inline void* addr(){return (void*)this;}                                                      \
        constexpr inline const void* addr()const{return (void*)this;}                                           \
        constexpr inline size_t children() const{return 1;}                                                     \
        using impl_base::NAME<A>::NAME;                                                                         \
        using typename impl_base::NAME<A>::base;                                                                \
    };                                                                                                          \
    template<typename A>                                                                                        \
    fields_t  NAME <A> :: fields(const path_t* steps)const {                                                    \
        if(steps[0]==END){                                                                                      \
            return this->fields();                                                                              \
        }                                                                                                       \
        else{                                                                                                   \
            if(steps[0]==LEFT){return this->left().fields(steps+1);}                                            \
        }                                                                                                       \
        return {nullptr,0};                                                                                     \
    }                                                                                                           \
    template<typename A>                                                                                        \
    constexpr bool NAME <A> :: tree_visit_pre(const visitor_t& op){                                             \
        auto sval = op(this->name(),this->fields(), this->addr(), this->children());                            \
        auto lval = this->left().tree_visit_pre(op);                                                            \
        return sval && lval;                                                                                    \
    }                                                                                                           \
    template<typename A>                                                                                        \
    constexpr bool NAME <A> :: tree_visit_post(const visitor_t& op){                                            \
        auto lval = this->left().tree_visit_post(op);                                                           \
        auto sval = op(this->name(),this->fields(), this->addr(), this->children());                            \
        return sval && lval;                                                                                    \
    }                                                                                                           \
    template<typename A>                                                                                        \
    constexpr bool NAME <A> :: ctree_visit_pre(const cvisitor_t& op) const{                                     \
        auto sval = op(this->name(),this->fields(), this->addr(), this->children());                            \
        auto lval = this->left().ctree_visit_pre(op);                                                           \
        return sval && lval;                                                                                    \
    }                                                                                                           \
    template<typename A>                                                                                        \
    constexpr bool NAME <A> :: ctree_visit_post(const cvisitor_t& op) const{                                    \
        auto lval = this->left().ctree_visit_post(op);                                                          \
        auto sval = op(this->name(),this->fields(), this->addr(), this->children());                            \
        return sval && lval;                                                                                    \
    }                                                                                                           \
    template<typename A>                                                                                        \
    uint64_t NAME <A> :: to_tree(tree::builder& dst)const {                                                     \
        auto lname= base::left().to_tree(dst);                                                                  \
        if constexpr(std::is_same<typename base::cfg_t, utils::empty_t>()){                                     \
            NAME <utils::tree_idx_ref<utils::tree_idx<A>>> tmp({(uint16_t)(dst.next()-lname)});                 \
            auto ret = dst.push(tree::op_t:: NAME, (uint8_t*)&tmp, sizeof(decltype(tmp)));                      \
            return ret;                                                                                         \
        }                                                                                                       \
        else{                                                                                                   \
            NAME <utils::tree_idx_ref<utils::tree_idx<A>>> tmp({(uint16_t)(dst.next()-lname)}, this->cfg);      \
            auto ret = dst.push(tree::op_t:: NAME, (uint8_t*)&tmp, sizeof(decltype(tmp)));                      \
            return ret;                                                                                         \
        }                                                                                                       \
    }                                                                                                           \
}}                                                                                                              \
namespace comptime {                                                                                            \
    constexpr inline auto NAME(auto&& a){                                                                       \
        using left_t = std::remove_reference<decltype(a)>::type;                                                \
        static_assert(std::derived_from<impl:: NAME <left_t>, utils::unary_op<left_t>>, "Operator must be derived from unary_op");\
        return impl::NAME <left_t>(a);                                                                          \
    };                                                                                                          \
    constexpr inline auto NAME(auto&& a, const typename impl::NAME <typename std::remove_reference<decltype(a)>::type>::base::cfg_t & cfg){                                  \
        using left_t = std::remove_reference<decltype(a)>::type;                                                \
        static_assert(std::derived_from<impl:: NAME <left_t>, utils::unary_op<left_t,typename impl::NAME <typename std::remove_reference<decltype(a)>::type>::base::cfg_t>>, "Operator must be derived from unary_op");\
        return impl::NAME <left_t>(a,cfg);                                                                      \
    };                                                                                                          \
}                                                                                                               \
namespace polymorphic {                                                                                         \
    template<typename Attrs=default_attrs>                                                                      \
    constexpr inline auto NAME(auto&& a){                                                                       \
        using left_t = std::remove_reference<decltype(a)>::type;                                                \
        return utils::dyn_op<Attrs,impl::NAME<left_t>>(a);                                                      \
    };                                                                                                          \
    template<typename Attrs=default_attrs>                                                                      \
    constexpr inline auto NAME(auto&& a, const typename impl::NAME <typename std::remove_reference<decltype(a)>::type>::base::cfg_t & cfg){                                   \
        using left_t = std::remove_reference<decltype(a)>::type;                                                \
        static_assert(std::derived_from<impl:: NAME <left_t>, utils::unary_op<left_t,typename impl::NAME <typename std::remove_reference<decltype(a)>::type>::base::cfg_t>>, "Operator must be derived from unary_op");\
        return utils::dyn_op<Attrs,impl::NAME <left_t>>(a,cfg);                                                 \
    };                                                                                                          \
}                                                                                                               \
namespace dynamic {                                                                                             \
    template<typename Attrs=default_attrs>                                                                      \
    using NAME##_t=utils::dyn_op<Attrs,impl::NAME<std::shared_ptr<utils::base_dyn<Attrs>>>>;                    \
    template<typename Attrs=default_attrs>                                                                      \
    constexpr inline std::shared_ptr<utils::base_dyn<Attrs>> NAME                                               \
    (                                                                                                           \
        std::shared_ptr<utils::base_dyn<Attrs>> a                                                               \
    ){                                                                                                          \
        return std::make_shared<utils::dyn_op<Attrs,impl::NAME<decltype(a)>>>(a);                               \
    }                                                                                                           \
    template<typename Attrs=default_attrs>                                                                      \
    constexpr inline std::shared_ptr<utils::base_dyn<Attrs>> NAME                                               \
    (                                                                                                           \
        std::shared_ptr<utils::base_dyn<Attrs>> a,                                                              \
        const typename impl::NAME<std::shared_ptr<utils::base_dyn<Attrs>>>::base::cfg_t& cfg                    \
    ){                                                                                                          \
        return std::make_shared<utils::dyn_op<Attrs,impl::NAME<decltype(a)>>>(a,cfg);                           \
    }                                                                                                           \
}  


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/// Used to record a binary operator `NAME` across all namespaces and an operator overload for it
#define sdf_register_operator_2o(NAME, OP)                                                                      \
sdf_register_operator_2(NAME)                                                                                   \
namespace comptime {                                                                                            \
    template <typename A, typename B> requires sdf_i<A> && sdf_i<B>                                             \
    constexpr inline auto operator OP (A a, B b){return NAME(a,b);}                                             \
}                                                                                                               \
namespace polymorphic {                                                                                         \
    template <typename A, typename B> requires sdf_i<A> && sdf_i<B>                                             \
    constexpr inline auto operator OP (A a, B b){return NAME(a,b);}                                             \
}                                                                                                               \
namespace dynamic {                                                                                             \
    template <typename A, typename B> requires sdf_i<A> && sdf_i<B>                                             \
    constexpr inline auto operator OP (std::shared_ptr<A> a, std::shared_ptr<B> b){return NAME(a,b);}           \
}                                                                                                               \


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


/// Used to record a unary operator `NAME` across all namespaces and an operator overload for it
#define sdf_register_operator_1o(NAME, OP)                                                                      \
sdf_register_operator_1(NAME)                                                                                   \
namespace comptime {                                                                                            \
    template <typename A> requires sdf_i<A>                                                                     \
    constexpr inline auto operator OP (A a){return NAME(a);}                                                    \
}                                                                                                               \
namespace polymorphic {                                                                                         \
    template <typename T> requires sdf_i<A>                                                                     \
    constexpr inline auto operator OP (A a){return NAME(a);}                                                    \
}                                                                                                               \
namespace dynamic {                                                                                             \
    template <typename A> requires sdf_i<A>                                                                     \
    constexpr inline auto operator OP (std::shared_ptr<A> a){return NAME(a);}                                   \
}                                                                                                               \


#define FIELD(PARENT, TYPE, WIDGET, NAME, DESC, MIN, MAX, DEF, VALIDATE) \
    {false,field_t::type_##TYPE, field_t::widget_##WIDGET, #NAME, DESC, offsetof(PARENT,NAME), sizeof(NAME), SVal< MIN>, SVal< MAX >, SVal< DEF >, (bool(*)(const void* value))VALIDATE}

#define FIELD_R(PARENT, TYPE, WIDGET, NAME, DESC) \
    {false,field_t::type_##TYPE, field_t::widget_##WIDGET, #NAME, DESC, offsetof(PARENT,NAME), sizeof(NAME), nullptr, nullptr, SVal<TYPE{}>, nullptr}

#define RO_FIELD(PARENT, TYPE, WIDGET, NAME, DESC, MIN, MAX, DEF, VALIDATE) \
    {true,field_t::type_##TYPE, field_t::widget_##WIDGET, #NAME, DESC, offsetof(PARENT,NAME), sizeof(NAME), SVal< MIN >, SVal< MAX >, SVal< DEF >, (bool(*)(const void* value))VALIDATE}

#define RO_FIELD_R(PARENT, TYPE, WIDGET, NAME, DESC) \
    {true,field_t::type_##TYPE, field_t::widget_##WIDGET, #NAME, DESC, offsetof(PARENT,NAME), sizeof(NAME), nullptr, nullptr, SVal<TYPE{}>, nullptr}

#define FIELD_OP(PARENT, TYPE, WIDGET, NAME, DESC, MIN, MAX, DEF, VALIDATE) \
    {false,field_t::type_##TYPE, #NAME, field_t::widget_##WIDGET, DESC, /*offsetof(PARENT, cfg)+*/offsetof(typename base::cfg_t, NAME) , sizeof(base::cfg.NAME), SVal< MIN >, SVal< MAX >, SVal< DEF >, (bool(*)(const void* value))VALIDATE}

#define FIELD_OP_R(PARENT, TYPE, WIDGET, NAME, DESC) \
    {false,field_t::type_##TYPE, field_t::widget_##WIDGET, #NAME, DESC, /*offsetof(PARENT, cfg)+*/offsetof(typename base::cfg_t, NAME) , sizeof(base::cfg.NAME), nullptr, nullptr, SVal<TYPE{}>, nullptr}

#define PRIMITIVE_NORMAL \
    constexpr inline const char* name()const{return _name;}\
    constexpr inline fields_t fields()const{return {_fields,sizeof(_fields)/sizeof(field_t)};}\
    constexpr inline visibility_t is_visible() const{return visibility_t::VISIBLE;}\
    constexpr inline vec3 normals(const glm::vec3& pos)const {\
        float d = sample(pos);\
        constexpr vec2 e = {EPS/2.0,0};\
        return normalize(d-vec3(\
            sample(pos-glm::vec3{e.x,e.y,e.y}),\
            sample(pos-glm::vec3{e.y,e.x,e.y}),\
            sample(pos-glm::vec3{e.y,e.y,e.x})\
        ));\
    }

//TODO: the split between primitive_normal and commons is wrong, it should be reshaped a bit.

#define PRIMITIVE_COMMONS \
    constexpr inline attrs_t operator()(const glm::vec3& pos)const {\
        float tmp=sample(pos); \
        return {tmp,normals(pos),tmp<MIX_EPS?cfg:typename attrs_t::extras_t{}};\
    }\
    PRIMITIVE_NORMAL

#define PRIMITIVE_TRAIT_SYM to.is_sym={true,true,true};
#define PRIMITIVE_TRAIT_GOOD to.is_exact_inner=true;to.is_exact_outer=true;to.is_bounded_inner=true;to.is_bounded_outer=true;

/* #endregion */

/* #region sdf-components */

//Basic 2D/3D primitives
#include "primitives/zero.hpp"
#include "primitives/demo.hpp"
#include "primitives/plane.hpp"
#include "primitives/sphere.hpp"
#include "primitives/box.hpp"
//#include "primitives/frame.hpp"
//#include "primitives/cylinder.hpp"
//#include "primitives/cone.hpp"
//#include "primitives/sketch2d.hpp"

#include "modifiers/forward.hpp"
#include "modifiers/optional.hpp"
#include "modifiers/located.hpp"
#include "modifiers/material.hpp"
#include "modifiers/lod.hpp"

//Boolean operators
#include "operators/boolean/join.hpp"
#include "operators/boolean/common.hpp"
#include "operators/boolean/cut.hpp"
#include "operators/boolean/xor.hpp"

//Smooth booleans
#include "operators/s-boolean/smooth-join.hpp"
#include "operators/s-boolean/smooth-common.hpp"
#include "operators/s-boolean/smooth-cut.hpp"

//Basic transforms
#include "operators/translate.hpp"
#include "operators/rotate.hpp"
#include "operators/scale.hpp"

//TODO: this might be unlocked for not host targets if the global buffers are used to store the actual pointers as it was done in the following data structures.
#if SDF_IS_HOST==true
#include "special/dynlib.hpp"
#endif
#include "special/interpreted.hpp"
#include "special/octa-sampled-3d.hpp"
#include "special/octa-sampled-2d.hpp"


//Construction
//#include "operators/extrude.hpp"
//#include "operators/revolve.hpp"
//#include "operators/array.hpp"
//#include "operators/cyclic-array.hpp"
//#include "operators/elongate.hpp"
//#include "operators/round.hpp"
//#include "operators/shell.hpp"
//#include "operators/mirror.hpp"

/* #endregion */

/* #region cleanup-preprocessor */

#undef SDF_INTERNALS


#undef sdf_register_primitive
#undef sdf_register_primitive_ostream
#undef sdf_register_primitive_xml
#undef sdf_register_operator_2
#undef sdf_register_operator_2_ostream
#undef sdf_register_operator_2_xml
#undef sdf_register_operator_1
#undef sdf_register_operator_1_ostream
#undef sdf_register_operator_1_xml
#undef sdf_register_operator_2o
#undef sdf_register_operator_1o

#undef FIELD
#undef FIELD_R
#undef FIELD_OP
#undef FIELD_OP_R
#undef RO_FIELD
#undef RO_FIELD_R
#undef FIELD_BASE

#undef PRIMITIVE_NORMAL
#undef PRIMITIVE_COMMONS
#undef PRIMITIVE_TRAIT_GOOD
#undef PRIMITIVE_TRAIT_SYM

/* #endregion */

/* #region tree_idx_impl */

#define SDF_TREE_DISPATCH_PRIMITIVE(OPCODE, OPERATION, RET) \
case tree::op_t:: OPCODE: {\
    impl:: OPCODE <Attrs>& ref= *(impl:: OPCODE <Attrs>*)((uint8_t*)this);\
    RET ref. OPERATION ;\
    break;\
}

#define SDF_TREE_DISPATCH_OPERATOR2(OPCODE, OPERATION, RET)\
case tree::op_t:: OPCODE : {\
    impl:: OPCODE <tree_idx_ref<tree_idx<Attrs>>,tree_idx_ref<tree_idx<Attrs>>>& ref= *(impl:: OPCODE <tree_idx_ref<tree_idx<Attrs>>,tree_idx_ref<tree_idx<Attrs>>>*)((uint8_t*)this);\
    RET ref. OPERATION ;\
    break;\
}

#define SDF_TREE_DISPATCH_OPERATOR1(OPCODE, OPERATION, RET)\
case tree::op_t:: OPCODE : {\
    impl:: OPCODE <tree_idx_ref<tree_idx<Attrs>>>& ref= *(impl:: OPCODE <tree_idx_ref<tree_idx<Attrs>>>*)((uint8_t*)this);\
    RET ref. OPERATION ;\
    break;\
}

#define SDF_TREE_DISPATCH(OPERATION, RET) \
switch(*(sdf::tree::op_t::type_t*)((uint8_t*)this-2)){\
    SDF_TREE_DISPATCH_PRIMITIVE(Sphere, OPERATION, RET) \
    SDF_TREE_DISPATCH_PRIMITIVE(Box, OPERATION, RET) \
    SDF_TREE_DISPATCH_PRIMITIVE(Plane, OPERATION, RET) \
    SDF_TREE_DISPATCH_PRIMITIVE(Zero, OPERATION, RET) \
    \
    SDF_TREE_DISPATCH_OPERATOR2(Join, OPERATION, RET) \
    SDF_TREE_DISPATCH_OPERATOR2(Xor, OPERATION, RET) \
    SDF_TREE_DISPATCH_OPERATOR2(Cut, OPERATION, RET) \
    SDF_TREE_DISPATCH_OPERATOR2(Common, OPERATION, RET) \
    SDF_TREE_DISPATCH_OPERATOR1(Translate, OPERATION, RET) \
    SDF_TREE_DISPATCH_OPERATOR1(Rotate, OPERATION, RET) \
    SDF_TREE_DISPATCH_OPERATOR1(Scale, OPERATION, RET) \
    SDF_TREE_DISPATCH_OPERATOR2(SmoothJoin, OPERATION, RET) \
    default:\
    {\
        printf("Fuck you! %d %p %p",*(sdf::tree::op_t::type_t*)((uint8_t*)this-2),(void*)((uint8_t*)this-2), (const void*)this );\
        RET {};\
        break;\
    }\
}


namespace sdf{
namespace utils{
    template <typename Attrs>
    inline float tree_idx<Attrs>::sample(const glm::vec3& pos) const{
        SDF_TREE_DISPATCH(sample(pos),return);
        return {};
    }

    template <typename Attrs>
    inline Attrs tree_idx<Attrs>::operator()(const glm::vec3& pos) const{
        //printf("[dispatch] %d, %d\n", (sdf::tree::op_t::type_t)*(uint16_t*)((uint8_t*)base+offset-2),offset);
        SDF_TREE_DISPATCH(operator()(pos),return);
        return {};
    }

    template<typename Attrs>
    inline const char* tree_idx<Attrs>::name() const{
        SDF_TREE_DISPATCH(name(),return);
        return "UNKNOWN";
    }

    template<typename Attrs>
    inline fields_t tree_idx<Attrs>::fields() const{
        SDF_TREE_DISPATCH(fields(),return);
        return {nullptr,0};
    }

    template<typename Attrs>
    inline fields_t tree_idx<Attrs>::fields(const path_t* steps) const{
        SDF_TREE_DISPATCH(fields(steps),return);
        return {nullptr, 0};
    }

    template <typename Attrs>
    inline bool tree_idx<Attrs>::tree_visit_pre(const visitor_t& op){
        SDF_TREE_DISPATCH(tree_visit_pre(op),return);
        return false;
    }

    template <typename Attrs>
    inline bool tree_idx<Attrs>::tree_visit_post(const visitor_t& op){
        SDF_TREE_DISPATCH(tree_visit_post(op),return);
        return false;
    }

    template <typename Attrs>
    inline bool tree_idx<Attrs>::ctree_visit_pre(const cvisitor_t& op)const{
        SDF_TREE_DISPATCH(ctree_visit_pre(op),return);
        return false;
    }

    template <typename Attrs>
    inline bool tree_idx<Attrs>::ctree_visit_post(const cvisitor_t& op)const{
        SDF_TREE_DISPATCH(ctree_visit_post(op),return);
        return false;
    }

    template <typename Attrs>
    inline size_t tree_idx<Attrs>::children() const{
        SDF_TREE_DISPATCH(children(),return);
        return false;
    }

    template <typename Attrs>
    inline void* tree_idx<Attrs>::addr(){
        SDF_TREE_DISPATCH(addr(),return);
        return nullptr;
    }

    template <typename Attrs>
    inline const void* tree_idx<Attrs>::addr()const{
        SDF_TREE_DISPATCH(addr(),return);
        return nullptr;
    }

    template <typename Attrs>
    inline void tree_idx<Attrs>::traits(traits_t& dst) const{
        SDF_TREE_DISPATCH(traits(dst),);
        return;
    }
}
}

#undef SDF_TREE_DISPATCH_PRIMITIVE
#undef SDF_TREE_DISPATCH_OPERATOR2
#undef SDF_TREE_DISPATCH_OPERATOR1
#undef SDF_TREE_DISPATCH


/* #endregion Implementation for the tree_idx logic binding underlying functions */

