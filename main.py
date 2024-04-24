import pandas as pd
import numpy as np
import requests
import logging
import json

# Set up logging
logger = logging.getLogger(__name__)

class Processor:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    # Function loads the data from json file
    def load_data(self,
                  url,
                  df_name):
        ''' 
        Args:
        url: String -> URL to extract file, 
        df_name: DataFrame -> extract the nested dictionary
                            to retrieve the content
        Returns:
        data: DataFrame -> laoded dataframe with original data
        '''
        try:
            x = requests.get(url)
            x.raise_for_status()
            dict1 = json.loads(x.text)
            data = pd.DataFrame(dict1[df_name])
            self.logger.info(f"Data loaded successfully from {url}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
        return data

    #Extracts the mentioned columns in the list of columns
    def select_columns(self,
                       df_name,
                       column_name,
                       output_df):
        '''
        Args:
        df_name: DataFrame -> Original data containing dataframe

        column_name: List -> Particular column name that needs 
                            to be selected and replicated in 
                            another DataFrame
        Returns:
        output_df: DataFrame -> Resulting/temporary dataframe to
                                make preprocessing
        '''
        try:
            output_df[column_name] = df_name[column_name]
            self.logger.info(f"Columns {column_name} successfully added to output DataFrame.")
        except KeyError as e:
            self.logger.error(f"Column(s) {column_name} not found in DataFrame: {e}")
        return output_df

    #Concatenates two columns in a DataFrame with a specified expression and stores the result in a new column of another DataFrame.
    def concat_columns(self,
                       df_name,
                       col1,
                       col2,
                       exp,
                       newcol,
                       output_df):
        '''
        Args:
        df_name: DataFrame -> Original data containing dataframe

        col1: String -> First column name needed to be concatenated

        col2: String -> Second column name needed to be concatenated

        exp: String -> Expression that needs to be added " " or ",:;/?"

        newcol: String -> Specifies the name of the new column of the resulting concatenation

        Returns:
        output_df: DataFrame -> Resulting/temporary dataframe to make preprocessing
        '''
        output_df[newcol] = ''
        try:
            for idx, row in df_name.iterrows():
                if pd.isna(row[col1]) or row[col1] == '':
                    output_df.at[idx, newcol] = 'Default Value'
                elif row[col1] == 'org':
                    output_df.at[idx, newcol] = row[col1]
                else:
                    second_part = row[col2] if pd.notna(row[col2]) and row[col2] != '' else ''
                    output_df.at[idx,newcol]=row[col1]+exp+second_part
                self.logger.info(f"Column {newcol} created successfully.")
        except Exception as e:
            self.logger.error(f"Error while concatenating columns: {e}")
        return output_df

    #Extracts and aggregates unique prize years and categories from a DataFrame containing prize details.
    def select_unique_prizes(self,df):
        '''
        Args:
        df: DataFrame -> The dataframe from which to extract prize information.

        Returns:
        (): Tuple -> A tuple of two lists containing unique prizes
        '''
        unique_year = []
        unique_category = []
        try:
            for row in df['prizes']:
                row_year = set()
                row_cat = set()
                for i in row:
                    year = i['year']
                    if year not in row_year:
                        row_year.add(year)
                    category = i['category']
                    if category not in row_cat:
                        row_cat.add(category)
                string_year = ";".join(row_year)
                unique_year.append(string_year)
                
                string_category = ";".join(row_cat)
                unique_category.append(string_category)
                self.logger.info("Successfully processed unique years and categories")
        except Exception as e:
            self.logger.error(f"Failed to process unique prizes: {e}")
        
        return (unique_year, unique_category)

    #Merge two dataframes with options for renaming columns, specifying merge key and method.
    def merge_dataframes(self,primary_df, 
                         secondary_df, 
                         rename_dict, 
                         merge_key, 
                         how='inner', 
                         copy=False):
        '''
        Args:
        primary_df: DataFrame -> Primary DataFrame to merge.

        secondary_df: DataFrame -> Secondary DataFrame to merge.

        rename_dict: dict -> Dictionary to rename columns in the secondary DataFrame.

        merge_key: str -> Column name on which to merge the DataFrames.

        how: str(Default Value=left) -> Type of merge to perform.

        copy: bool(Default Value=false) -> Whether to copy data from secondary DataFrame.

        Returns: 
        merged_df: DataFrame -> A merged DataFrame as per the specified parameters.
        '''
        try:
            if rename_dict:
                secondary_df = secondary_df.rename(columns=rename_dict)
            merged_df = primary_df.merge(secondary_df, 
                                         how=how, 
                                         on=merge_key, 
                                         copy=copy)
            self.logger.info("DataFrames merged successfully")
            return merged_df
        except Exception as e:
            self.logger.error(f"An error occurred during merging: {e}")
            return None

    #Save dataframes in csv file specified
    def save_csv(self,df_name, file_name):
        '''
        Args:
        df_name: DataFrame -> dataframe containing final output columns

        file_name: String -> filename of csv file to be saved in.
        '''
        try:
            df_name.to_csv(file_name)
            self.logger.info(f"DataFrame was successfully saved to {file_name}")
            return True
        except PermissionError:
            self.logger.error(f"Permission denied: Check file permissions or if the file is open elsewhere.")
            return False
        except FileNotFoundError:
            self.logger.error(f"The directory for {file_name} does not exist. Ensure the directory path is correct.")
            return False
        except Exception as e:
            self.logger.error(f"Failed to save DataFrame to {file_name}: {e}")
            raise e
        
def main():
    #Creating a instance of the class created necessary to call the functions
    dfobj=Processor()

    #Necessary to activate logging status on a file.
    logging.basicConfig(filename='assignment.log', level=logging.INFO)
    logger.info('Started')

    #Calling load_data function from class object created and loading the json file content on dataframe
    laureates_data = dfobj.load_data("https://raw.githubusercontent.com/ashwath92/nobel_json/main/laureates.json","laureates")
    country_data = dfobj.load_data("https://raw.githubusercontent.com/ashwath92/nobel_json/main/countries.json",'countries')
    
    #Saving the original data for insights
    dfobj.save_csv(laureates_data,'laureates_data.csv')
    dfobj.save_csv(country_data,'country_data.csv')

    #Creating temporary and final dataframes
    combined_df = pd.DataFrame()
    result_df = pd.DataFrame()

    # TASK 1: adding id(laureates df) column to the dataframe
    dfobj.select_columns(laureates_data,'id',combined_df)


    # TASK 2: concatenating 2 columns with an expression
    dfobj.concat_columns(laureates_data,'firstname','surname',' ','name',combined_df)


    # TASK 3: dob that is born(laureates df) to another df
    dfobj.select_columns(laureates_data,'born',combined_df)
    combined_df.rename(columns={'born':'dob'},inplace=True)


    # TASK 4 and TASK 5: unique_prize_years & unique_prize_categories (concat all unique years and categories in the 'prizes' field using ;)
    t4, t5 = dfobj.select_unique_prizes(laureates_data)

    task4 = pd.DataFrame(t4)
    task5 = pd.DataFrame(t5)

    combined_df['unique_prize_year'] = task4
    combined_df['unique_prize_category'] = task5


    # TASK 6 adding gender column in the df
    dfobj.select_columns(laureates_data,'gender',combined_df)


    # TASK 7 
    combined_df = combined_df.merge(laureates_data)

    dfobj.merge_dataframes(combined_df, 
                           country_data, 
                           {'code':'bornCountryCode',
                            'name':'Country_name',}, 
                           'bornCountryCode', 
                           how='inner', copy=False)
    
    # Selecting and adding final processed df from temporary to final df using 
    dfobj.select_columns(combined_df,
                         ['id','name','dob',
                          'unique_prize_year','unique_prize_category','gender',
                          'bornCountryCode',],
                         result_df)

    dfobj.save_csv(result_df,'result_data.csv')

    logger.info('Finished')


if __name__ == '__main__':
    main()
