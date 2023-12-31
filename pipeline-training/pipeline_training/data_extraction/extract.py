"""
NOTE:
1. Liskov Substitution Principle (LSP): Although not directly implemented in
   this module, the design does consider LSP. The 'extract_from_data_warehouse'
   function accepts any 'Connection' object and uses its 'query' method. As a
   result, any subclass of 'Connection' with a 'query' method could be passed
   to this function without affecting program correctness. Note that a direct
   implementation would involve creating specific subclasses of 'Connection'
   for different data warehouse systems.

    NOTE:
        Of course this is not the case since I only have implemented a wrapper
        for BigQuery, and BigQuery is not even a subclass of Connection. But
        the principle is still relevant.
"""

import pandas as pd
from common_utils.core.base import Connection
from common_utils.core.logger import Logger

from conf.base import Config
from metadata.core import Metadata


class Extract:
    def __init__(
        self, cfg: Config, logger: Logger, connection: Connection, metadata: Metadata
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.connection = connection
        self.metadata = metadata

    def extract_from_data_warehouse(self, query: str) -> Metadata:
        """
        Extracts data from a data warehouse and updates metadata. This function
        reflects several principles of good software design:

        1. Single Responsibility Principle (SRP): Adheres to SRP with a single
        responsibility of extracting data and updating metadata.

        2. Liskov Substitution Principle (LSP): Designed with LSP in mind by
        accepting any object as 'connection' which has a 'query' method.

        3. Dependency Inversion Principle (DIP): The function adheres to DIP by having
        the high-level 'extract_from_data_warehouse' function depend on the
        abstraction (the 'Connection' interface), not the details of any specific
        connection class. This principle is also demonstrated through the use of
        Dependency Injection, where dependencies (like the connection, logger, and
        metadata) are passed as arguments to the function rather than being
        hardcoded or tightly coupled to the function implementation.

        4. Interface Segregation Principle (ISP): Uses 'Connection' class as
        an interface, enforcing that any connection object must have a
        'query' method.

        NOTE:
            1. The caller is responsible for creating and configuring the
            'Connection', 'Logger', and 'Metadata' objects.
            2. This is a function to be called in development.

        Parameters
        ----------
        connection : Connection
            Object implementing a 'query' method to fetch data from a database.
        query : str
            SQL query to fetch data from the database.
        logger : Logger
            Logger object for logging information and errors.
        metadata : Metadata
            Metadata object where fetched data and relevant attributes will be stored.

        Returns
        -------
        metadata : Metadata
            Updated metadata with attributes set from the fetched data.
        """
        self.logger.info("Development Environment: starting data extraction...")

        try:
            # assuming that the connection object has a `query` method
            raw_df: pd.DataFrame = self.connection.query(query)
            self.logger.info("✅ Data extraction completed. Updating metadata...")

            num_rows, num_cols = raw_df.shape
            dataset, table_name = self.connection.dataset, self.connection.table_name

            attr_dict = {
                "raw_df": raw_df,
                "raw_num_rows": num_rows,
                "raw_num_cols": num_cols,
                "raw_dataset": dataset,
                "raw_table_name": table_name,
                "raw_query": query,
            }

            self.metadata.set_attrs(attr_dict)
            return self.metadata
        except Exception as error:
            self.logger.error(f"❌ Data extraction failed. Error: {error}")
            raise error

    def run(self) -> Metadata:
        """
        Runs the data extraction pipeline.

        Returns
        -------
        metadata : Metadata
            Updated metadata with attributes set from the fetched data.
        """
        return self.extract_from_data_warehouse(
            **self.cfg.extract.extract_from_data_warehouse.model_dump(mode="python")
        )
