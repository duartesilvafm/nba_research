import pandas as pd
import plotly.express as px

'''
TEAM CLASS - process and clean data to get team level data
'''

class Team():


    @staticmethod
    def get_assist_ratio(data, ast, fga, fta, to):
        '''
        Method to calculate a teams assists ratio
        '''

        numerator = data[ast] * 100
        denominator = data[fga] + (0.44 * data[fta]) + data[ast] + data[to]

        return numerator / denominator


    def __init__(self, raw_data: pd.DataFrame):
        self.raw_data = raw_data
        self.data = self.raw_data.copy(deep = True)


    def get_game_data(self, game_id: str, groupby_columns: list, aggregator: dict):
        '''
        Method to get game-level data

        game_id: column to group by game_id
        groupby_columns: other columns to groupby (excluding game id h)
        agg: aggregator dictionary to aggregate data
        '''

        # group by columns
        self.game_data =  self.data.groupby(by = [game_id] + groupby_columns, as_index = False, dropna = False).agg(aggregator)


    def get_winner(self, column: str = 'Win', own_score: str = 'Team_Score', opponent_score: str = 'Opponent_Score'):
        '''
        Method to create helper dataframe to get whether the team won or lost
        
        column: name of the column to the bool values
        own_score: name of the column with teams_score
        opponent_score: name of the column with the other teams score
        '''

        # check if score columns are in the data
        assert (own_score and opponent_score) in self.data.columns, f'{own_score} or {opponent_score} not in columns'

        # create winner column
        self.data[column] = self.data.apply(lambda row: 1 if row[own_score] > row[opponent_score] else 0, axis = 1)


    def get_conversion_variable(self, makes: str, attempts: str):
        '''
        Method to calculate the % of made fgs, fts, etc.

        makes: variable for the makes
        attempts: variables for the attempts
        '''

        return self.data[makes] / self.data[attempts]


    def create_box_plot(self, data: pd.DataFrame, params_group: dict, params_bx: dict, params_sort:dict=None, sort:bool=False):
        '''
        Method to create box plot

        data: data to create box plot from
        id_vars: id_vars to keep for data melting
        value_vars: value_vars to keep for data melting
        y: variable to plot in the box plot
        var_name: var_name for melting dataframe
        x: variable for x axis in box plot
        color: variable to segregate data in boxplot
        '''

        # get data for box plot
        bx_data = self.group_for_graph(data=data, **params_group)
        
        if sort:
            bx_data = bx_data.sort_values(**params_sort)

        # create box plot diagram        
        fig = px.box(bx_data, **params_bx)
        fig.show()


    def create_scatter(self, data: pd.DataFrame, params_group: dict, params_scatter: dict):
        '''
        Method to create scatter plot base on certain parameters
        
        data: dataframe with data
        params_group: params to feed into grouping dataframe
        params_scatter: parameters to feed into scatter
        '''
        pass


    def group_for_graph(self, data, id_vars: list, value_vars: list, var_name: str = None):
        '''
        Hidden method to clean data for box plot
        '''
        bx_data = data.melt(id_vars=id_vars, value_vars=value_vars, var_name = var_name) 
        return bx_data


    @property
    def raw_data(self):
        return self._raw_data

    @raw_data.setter
    def raw_data(self, value):
        self._raw_data = value


class TeamSeason(Team):


    def __init__(self, raw_data: pd.DataFrame):
        super().__init__(raw_data)


    def get_season_stats(self, seasons_vars = ['Team_Abbrev', 'season'], agg: dict = {'game_id': 'count', 'Win': sum}):
        '''
        Method to get season statistics

        INVARIANT: a game is represented by a game_date
        '''

        # set season data
        self.season_data = self.data.groupby(by = seasons_vars, as_index = False).agg(agg)

        # rename vars
        self.season_data = self.season_data.rename(columns = {'game_id': 'games', 'Win': 'wins'})

        # get variables for loses
        self.season_data['loses'] = self.season_data['games'] - self.season_data['wins']