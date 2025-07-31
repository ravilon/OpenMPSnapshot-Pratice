/*--------------------------------------------------------------------
  (C) Copyright 2006-2014 Barcelona Supercomputing Center
                          Centro Nacional de Supercomputacion
  
  This file is part of Mercurium C/C++ source-to-source compiler.
  
  See AUTHORS file in the top level directory for information
  regarding developers and contributors.
  
  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 3 of the License, or (at your option) any later version.
  
  Mercurium C/C++ source-to-source compiler is distributed in the hope
  that it will be useful, but WITHOUT ANY WARRANTY; without even the
  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the GNU Lesser General Public License for more
  details.
  
  You should have received a copy of the GNU Lesser General Public
  License along with Mercurium C/C++ source-to-source compiler; if
  not, write to the Free Software Foundation, Inc., 675 Mass Ave,
  Cambridge, MA 02139, USA.
--------------------------------------------------------------------*/




#ifndef TL_OMP_CORE_HPP
#define TL_OMP_CORE_HPP

#include <stack>

#include "tl-lexer.hpp"
#include "tl-nodecl.hpp"
#include "tl-compilerphase.hpp"
#include "tl-pragmasupport.hpp"

#include "tl-omp.hpp"
#include "tl-omp-reduction.hpp"
#include "tl-ompss-target.hpp"

namespace TL
{
    namespace OpenMP
    {
    	struct UDRParsedInfo 
		{
			Type type;
            Nodecl::NodeclBase combine_expression;
			Symbol in_symbol;
			Symbol out_symbol;

            UDRParsedInfo() : type(NULL), combine_expression(), in_symbol(NULL), out_symbol(NULL) {}
		};

        class Core : public TL::PragmaCustomCompilerPhase
        {
            public:
                typedef std::map<TL::Symbol, OpenMP::Reduction*> reduction_map_info_t;
                static reduction_map_info_t reduction_map_info;
            private:
                void parse_new_udr(const std::string& str);

                void register_omp_constructs();
                void register_oss_constructs();

                void bind_omp_constructs();
                void bind_oss_constructs();

                // Handler functions
#define DECL_DIRECTIVE(_directive, _name, _pred, _func_prefix) \
                void _func_prefix##_name##_handler_pre(TL::PragmaCustomDirective); \
                void _func_prefix##_name##_handler_post(TL::PragmaCustomDirective);

#define DECL_CONSTRUCT(_directive, _name, _pred, _func_prefix) \
                void _func_prefix##_name##_handler_pre(TL::PragmaCustomStatement); \
                void _func_prefix##_name##_handler_post(TL::PragmaCustomStatement); \
                void _func_prefix##_name##_handler_pre(TL::PragmaCustomDeclaration); \
                void _func_prefix##_name##_handler_post(TL::PragmaCustomDeclaration);

#define OMP_DIRECTIVE(_directive, _name, _pred) DECL_DIRECTIVE(_directive, _name, _pred, /*empty_prefix*/ )
#define OMP_CONSTRUCT(_directive, _name, _pred) DECL_CONSTRUCT(_directive, _name, _pred, /*empty_prefix*/)
#define OMP_CONSTRUCT_NOEND(_directive, _name, _pred) OMP_CONSTRUCT(_directive, _name, _pred)
#include "tl-omp-constructs.def"
                // Section is special
                OMP_CONSTRUCT("section", section, true)
#undef OMP_CONSTRUCT
#undef OMP_CONSTRUCT_NOEND
#undef OMP_DIRECTIVE

#define OSS_DIRECTIVE(_directive, _name, _pred) DECL_DIRECTIVE(_directive, _name, _pred, oss_)
#define OSS_CONSTRUCT(_directive, _name, _pred) DECL_CONSTRUCT(_directive, _name, _pred, oss_)
#define OSS_CONSTRUCT_NOEND(_directive, _name, _pred) OSS_CONSTRUCT(_directive, _name, _pred)
#include "tl-oss-constructs.def"
#undef OSS_CONSTRUCT
#undef OSS_CONSTRUCT_NOEND
#undef OSS_DIRECTIVE

#undef DECL_DIRECTIVE
#undef DECL_CONSTRUCT

                static bool _constructs_already_registered;
                static bool _reductions_already_registered;
                static bool _already_informed_new_ompss_copy_deps;

                std::shared_ptr<OpenMP::Info> _openmp_info;
                void omp_target_handler_pre(TL::PragmaCustomStatement ctr);
                void omp_target_handler_post(TL::PragmaCustomStatement ctr);

                std::shared_ptr<OmpSs::FunctionTaskSet> _function_task_set;
                std::stack<OmpSs::TargetContext> _target_context;
                // OmpSs-2 Assert string list
                std::shared_ptr<OmpSs::AssertInfo> _ompss_assert_info;

                void ompss_target_handler_pre(TL::PragmaCustomStatement ctr);
                void ompss_target_handler_post(TL::PragmaCustomStatement ctr);

                void ompss_target_handler_pre(TL::PragmaCustomDeclaration ctr);
                void ompss_target_handler_post(TL::PragmaCustomDeclaration ctr);

                void ompss_common_target_handler_pre(TL::PragmaCustomLine pragma_line,
                        OmpSs::TargetContext& target_ctx,
                        TL::Scope scope,
                        bool is_pragma_task);

                // This function handles the implements clause if it was
                // present in a target or task construct.
                void ompss_handle_implements_clause(
                        const OmpSs::TargetContext& target_ctx,
                        Symbol function_sym,
                        const locus_t* locus);

                void ompss_get_target_info(TL::PragmaCustomLine pragma_line,
                        DataEnvironment& data_environment);

                void task_function_handler_pre(TL::PragmaCustomDeclaration construct);
                void task_inline_handler_pre(TL::PragmaCustomStatement construct);

                void get_clause_symbols(
                        PragmaCustomClause clause,
                        const TL::ObjectList<TL::Symbol> &symbols_in_construct,
                        ObjectList<DataReference>& data_ref_list,
                        DataEnvironment& data_environment,
                        ObjectList<Symbol>& extra_symbols);

                Nodecl::NodeclBase update_clause(
                        Nodecl::NodeclBase clause,
                        Symbol function_symbol);

                ObjectList<Nodecl::NodeclBase> update_clauses(
                        const ObjectList<Nodecl::NodeclBase>& clauses,
                        Symbol function_symbol);

                void get_reduction_symbols(
                        TL::PragmaCustomLine construct,
                        PragmaCustomClause clause,
                        TL::Scope scope,
                        const TL::ObjectList<TL::Symbol> &symbols_in_construct,
                        ObjectList<ReductionSymbol>& sym_list,
                        const TL::Symbol &function_sym = TL::Symbol::invalid());

                void get_reduction_explicit_attributes(
                        TL::PragmaCustomLine construct,
                        Nodecl::NodeclBase statements,
                        DataEnvironment& data_environment,
                        ObjectList<Symbol>& extra_symbols);

                void get_data_explicit_attributes(
                        TL::PragmaCustomLine construct,
                        Nodecl::NodeclBase statements,
                        DataEnvironment& data_environment,
                        ObjectList<Symbol> &extra_symbols);

                void get_data_implicit_attributes(
                        TL::PragmaCustomStatement construct,
                        DataSharingAttribute default_data_attr,
                        DataEnvironment& data_environment,
                        bool there_is_default_clause);

                // Helper function to analyze a list of symbols
                // like if they where inside task body
                void get_data_implicit_attributes_of_nonlocal_symbols(
                        const ObjectList<Nodecl::Symbol> &nonlocal_symbols_occurrences,
                        DataEnvironment& data_environment,
                        DataSharingAttribute default_data_attr,
                        bool there_is_default_clause);

                void get_data_implicit_attributes_task(
                        TL::PragmaCustomStatement construct,
                        DataEnvironment& data_environment,
                        DataSharingAttribute default_data_attr,
                        bool there_is_default_clause);

                void get_data_extra_symbols(
                        DataEnvironment& data_environment,
                        const ObjectList<Symbol>& extra_symbols);

                void get_data_implicit_attributes_of_indirectly_accessible_symbols(
                        TL::PragmaCustomStatement construct,
                        DataEnvironment& data_environment,
                        ObjectList<TL::Symbol>& nonlocal_symbols);

                // This function handles the dependences of a task construct,
                // adding new information to the data environment
                void handle_task_dependences(
                        PragmaCustomLine pragma_line,
                        Nodecl::NodeclBase parsing_context,
                        DataSharingAttribute default_data_attr,
                        DataEnvironment& data_environment,
                        ObjectList<Symbol>& extra_symbols);

                // This function handles the dependences of a taskwait construct,
                // adding new information to the data environment
                void handle_taskwait_dependences(
                        PragmaCustomLine pragma_line,
                        Nodecl::NodeclBase parsing_context,
                        DataSharingAttribute default_data_attr,
                        DataEnvironment& data_environment,
                        ObjectList<Symbol>& extra_symbols);

                // This function is used to define a concurrent dependence over
                // the reduction expressions
                void handle_implicit_dependences_of_task_reductions(
                        PragmaCustomLine pragma_line,
                        DataSharingAttribute default_data_attr,
                        DataEnvironment& data_environment,
                        ObjectList<Symbol>& extra_symbols);

                // This function handles clauses of a task construct
                // like if they where in task body,
                // adding new information to the data environment
                void handle_task_body_like_clauses(
                        TL::PragmaCustomStatement construct,
                        DataEnvironment& data_environment,
                        DataSharingAttribute default_data_attr,
                        bool there_is_default_clause);

                // This function handles clauses that have one
                // expression
                void handle_clause_with_one_expression(
                        const TL::PragmaCustomStatement& directive,
                        const std::string &clause_name,
                        Nodecl::NodeclBase parsing_context,
                        DataEnvironment &data_environment,
                        void (DataEnvironment::*set_clause)(const Nodecl::NodeclBase&),
                        bool there_is_default_clause,
                        DataSharingAttribute default_data_attr);

                // This handles onready clause thaking into account
                // the special case of fortran
                void handle_onready_clause_expression(
                        const TL::PragmaCustomStatement& directive,
                        DataEnvironment &data_environment,
                        bool there_is_default_clause,
                        DataSharingAttribute default_data_attr);

                // Helper function that handles the basic set of dependences:
                // in, out and inout
                void get_basic_dependences_info(
                        PragmaCustomLine pragma_line,
                        Nodecl::NodeclBase parsing_context,
                        DataEnvironment& data_environment,
                        DataSharingAttribute default_data_attr,
                        ObjectList<Symbol>& extra_symbols);

                // Helper function that handles the OpenMP dependences clauses
                void get_dependences_openmp(
                        TL::PragmaCustomClause clause,
                        Nodecl::NodeclBase parsing_context,
                        DataEnvironment& data_environment,
                        DataSharingAttribute default_data_attr,
                        ObjectList<Symbol>& extra_symbols);

                ObjectList<Nodecl::NodeclBase> parse_dependences_ompss_clause(
                        PragmaCustomClause clause,
                        TL::ReferenceScope parsing_scope);

                void parse_dependences_openmp_clause(
                        TL::ReferenceScope parsing_scope,
                        TL::PragmaCustomClause clause,
                        TL::ObjectList<Nodecl::NodeclBase> &in,
                        TL::ObjectList<Nodecl::NodeclBase> &out,
                        TL::ObjectList<Nodecl::NodeclBase> &inout,
                        const locus_t* locus);

                DataSharingAttribute get_default_data_sharing(TL::PragmaCustomLine construct,
                        DataSharingAttribute fallback_data_sharing,
                        bool &there_is_default_clause,
                        bool allow_default_auto=false);

                void loop_handler_pre(TL::PragmaCustomStatement construct,
                        Nodecl::NodeclBase loop,
                        void (Core::*common_loop_handler)(TL::PragmaCustomStatement,
                            Nodecl::NodeclBase, DataEnvironment&, ObjectList<Symbol>&));

                void handle_map_clause(TL::PragmaCustomLine pragma_line,
                        DataEnvironment& data_environment);

                enum DefaultMapValue
                {
                    DEFAULTMAP_NONE = 0,
                    DEFAULTMAP_SCALAR = 1,
                };

                void compute_implicit_device_mappings(Nodecl::NodeclBase stmt,
                        DataEnvironment& data_environment,
                        DefaultMapValue);

                void common_parallel_handler(
                        TL::PragmaCustomStatement ctr,
                        DataEnvironment& data_environment,
                        ObjectList<Symbol>& extra_symbols);

                void common_for_handler(
                        TL::PragmaCustomStatement custom_statement,
                        Nodecl::NodeclBase nodecl,
                        DataEnvironment& data_environment,
                        ObjectList<Symbol>& extra_symbols);

                void common_while_handler(
                        TL::PragmaCustomStatement custom_statement,
                        Nodecl::NodeclBase statement,
                        DataEnvironment& data_environment,
                        ObjectList<Symbol>& extra_symbols);

                void common_construct_handler(
                        TL::PragmaCustomStatement construct,
                        DataEnvironment& data_environment,
                        ObjectList<Symbol>& extra_symbols);

                void common_workshare_handler(
                        TL::PragmaCustomStatement construct,
                        DataEnvironment& data_environment,
                        ObjectList<Symbol>& extra_symbols);

                void common_teams_handler(
                        TL::PragmaCustomStatement ctr,
                        DataEnvironment& data_environment,
                        ObjectList<Symbol>& extra_symbols);

                void fix_sections_layout(TL::PragmaCustomStatement construct, const std::string& pragma_name);

                void parse_declare_reduction(ReferenceScope ref_sc, const std::string& declare_reduction_src, bool is_builtin);
                void parse_declare_reduction(ReferenceScope ref_sc, Source declare_reduction_src, bool is_builtin);
                void parse_declare_reduction(ReferenceScope ref_sc,
                        const std::string &name,
                        const std::string &typenames,
                        const std::string &combiner,
                        const std::string &initializer);
                void parse_builtin_reduction(ReferenceScope ref_sc,
                        const std::string &name,
                        const std::string &typenames,
                        const std::string &combiner,
                        const std::string &initializer);

                void initialize_builtin_reductions(Scope sc);

                void sanity_check_for_loop(Nodecl::NodeclBase);

                bool _ompss_mode;
                bool _copy_deps_by_default;
                bool _untied_tasks_by_default;
                bool _discard_unused_data_sharings;
                bool _allow_shared_without_copies;
                bool _allow_array_reductions;
                bool _enable_input_by_value_dependences; // EXPERIMENTAL
                bool _enable_nonvoid_function_tasks;     // EXPERIMENTAL

                // States if we have seen a declare target
                bool _inside_declare_target;
            public:
                Core();

                virtual void run(TL::DTO& dto);
                virtual void pre_run(TL::DTO& dto);

                virtual void phase_cleanup(TL::DTO& data_flow);
                virtual void phase_cleanup_end_of_pipeline(TL::DTO& dto);

                virtual ~Core() { }

                std::shared_ptr<OpenMP::Info> get_openmp_info();

                //! Used when parsing declare reduction
                static bool _silent_declare_reduction;

                //! Returns whether the current programming model is OmpSs
                bool in_ompss_mode() const;

                //! Returns whether tasks should be untied by default
                bool untied_tasks_by_default() const;

                /* These methods are used from Base::Base() to parse the TL::Core phase flags */
                void set_ompss_mode_from_str(const std::string& str);
                void set_copy_deps_from_str(const std::string& str);
                void set_untied_tasks_by_default_from_str(const std::string& str);
                void set_discard_unused_data_sharings_from_str(const std::string& str);
                void set_allow_shared_without_copies_from_str(const std::string& str);
                void set_allow_array_reductions_from_str(const std::string& str);
                void set_enable_input_by_value_dependences_from_str(const std::string& str);
                void set_enable_nonvoid_function_tasks_from_str(const std::string& str);
        };

        Nodecl::NodeclBase get_statement_from_pragma(
                const TL::PragmaCustomStatement& construct);

        // OpenMP core is a one shot phase, so even if it is in the compiler
        // pipeline twice, it will only run once by default.
        // Call this function to reenable openmp_core. Use this function
        // when you are sure that your changes require a full OpenMP analysis
        void openmp_core_run_next_time(DTO& dto);
    }
}

#endif // TL_OMP_CORE_HPP
