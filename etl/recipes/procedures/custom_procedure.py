import logging
from typing import List

import pandas as pd

from ddf_utils.transformer import translate_column as tc
from ddf_utils.chef.helpers import debuggable, build_dictionary, read_opt
from ddf_utils.chef.model.ingredient import Ingredient, get_ingredient_class
from ddf_utils.chef.model.chef import Chef


logger = logging.getLogger('custom_procedure')


@debuggable
def translate_column_over_time(chef: Chef, ingredients: List[Ingredient],
                               result, dictionary, column, time_column) -> Ingredient:

    ingredient = ingredients[0]
    logger.info("translate_column_over_time: " + ingredient.id)
    di = ingredient.compute()
    mapping_df = chef.dag.node_dict[dictionary['base']].evaluate().compute()[dictionary['value']]

    new_data = dict()

    for k, df in di.items():
        new_data[k] = _run(df, mapping_df, column, time_column,
                           dictionary['key'], dictionary['value'])

    return get_ingredient_class(ingredient.dtype).from_procedure_result(
        result, ingredient.key, data_computed=new_data)


def _run(df, mapping_df, target_column, time_column, key, value):
    """translate one dataframe with the mapping_df, align by time_column"""
    g1 = df.groupby(time_column)
    g2 = mapping_df.groupby(time_column)

    new_dfs = list()

    for year, df_year in g1:
        try:
            mapping_g = g2.get_group(year)
        except KeyError:
            logger.warning(f"{year} not found in mapping dictionary, skipping")
            continue
        mapping_year = mapping_g.set_index(key)[value].to_dict()
        logger.debug('running on: ' + str(year))
        new_dfs.append(tc(df_year, target_column, 'inline', mapping_year, not_found='drop'))

    return pd.concat(new_dfs, ignore_index=True)


@debuggable
def population_percentage(chef, ingredients, result, world_population, align_column):
    ingredient = ingredients[0]
    df = ingredient.compute()['population']
    world_pop_df = chef.dag.node_dict[world_population].evaluate().compute()['population']

    world_pop_df = world_pop_df.set_index('year')

    grouped = df.groupby('income_groups')

    new_dfs = []

    for g, df_g in grouped:
        pop_percent = df_g.set_index('year')['population'] / world_pop_df['population'] * 100
        pop_percent = pop_percent.reset_index().dropna()
        pop_percent['income_groups'] = g
        pop_percent = pop_percent.rename(columns={'population': 'population_percentage'})
        new_dfs.append(pop_percent)

    new_data = dict(population_percentage=pd.concat(new_dfs, ignore_index=True))

    return get_ingredient_class(ingredient.dtype).from_procedure_result(
        result, ingredient.key, data_computed=new_data)
