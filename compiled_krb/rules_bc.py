# rules_bc.py

from pyke import contexts, pattern, bc_rule

pyke_version = '1.1.1'
compiler_version = 1

def table_legitimacy(rule, arg_patterns, arg_context):
  engine = rule.rule_base.engine
  patterns = rule.goal_arg_patterns()
  if len(arg_patterns) == len(patterns):
    context = contexts.bc_context(rule)
    try:
      if all(map(lambda pat, arg:
                   pat.match_pattern(context, context,
                                     arg, arg_context),
                 patterns,
                 arg_patterns)):
        rule.rule_base.num_bc_rules_matched += 1
        with engine.prove('facts', 'num_rows', context,
                          (rule.pattern(0),)) \
          as gen_1:
          for x_1 in gen_1:
            assert x_1 is None, \
              "rules.table_legitimacy: got unexpected plan from when clause 1"
            rule.rule_base.num_bc_rule_successes += 1
            yield
        rule.rule_base.num_bc_rule_failures += 1
    finally:
      context.done()

def table_empty(rule, arg_patterns, arg_context):
  engine = rule.rule_base.engine
  patterns = rule.goal_arg_patterns()
  if len(arg_patterns) == len(patterns):
    context = contexts.bc_context(rule)
    try:
      if all(map(lambda pat, arg:
                   pat.match_pattern(context, context,
                                     arg, arg_context),
                 patterns,
                 arg_patterns)):
        rule.rule_base.num_bc_rules_matched += 1
        with engine.prove('facts1', 'empty_values', context,
                          (rule.pattern(0),)) \
          as gen_1:
          for x_1 in gen_1:
            assert x_1 is None, \
              "rules.table_empty: got unexpected plan from when clause 1"
            rule.rule_base.num_bc_rule_successes += 1
            yield
        rule.rule_base.num_bc_rule_failures += 1
    finally:
      context.done()

def populate(engine):
  This_rule_base = engine.get_create('rules')
  
  bc_rule.bc_rule('table_legitimacy', This_rule_base, 'eligible',
                  table_legitimacy, None,
                  (pattern.pattern_literal("Il y a trop de ligne dans le modele"),),
                  (),
                  (pattern.pattern_literal('row_count'),))
  
  bc_rule.bc_rule('table_empty', This_rule_base, 'empty',
                  table_empty, None,
                  (pattern.pattern_literal("Il n y a pas d action immédiate prévue en cas de dommage"),),
                  (),
                  (pattern.pattern_literal('value'),))


Krb_filename = '..\\rules.krb'
Krb_lineno_map = (
    ((14, 18), (3, 3)),
    ((20, 25), (5, 5)),
    ((38, 42), (9, 9)),
    ((44, 49), (11, 11)),
)
