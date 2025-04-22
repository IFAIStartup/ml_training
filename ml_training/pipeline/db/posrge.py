"""IT IS NOT USED IN THIS REPOSITORY NOW. DO NOT TAKE IT INTO ACCOUNT"""

from sqlalchemy import Column, String, JSON
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy import create_engine, Engine


class Base(DeclarativeBase):
    pass


class Model(Base):
    __tablename__ = 'models'

    id = Column(String(100), primary_key=True, nullable=False)
    type = Column(String(100), nullable=False)
    weights_parent = Column(String(300), nullable=False)
    weights = Column(String(300), nullable=False, unique=True)
    data = Column(String(300), nullable=False)
    status = Column(String(100), nullable=False)
    logs = Column(JSON(none_as_null=True), nullable=False)
    worker = Column(String(100), nullable=False)
    
    def __repr__(self) -> str:
            repr =  f"Model(id={self.id!r}, " \
                    f"type={self.type!r}, "     \
                    f"weights_parent={self.weights_parent!r}, "   \
                    f"weights={self.weights!r}, "   \
                    f"data={self.data!r}, "  \
                    f"status={self.status!r}, "  \
                    f"logs={self.logs.keys()!r}" \
                    f"worker={self.worker!r})"

            return repr


class PostgreSQLHandler:
    __shared_engines = {}
    
    def __init__(
        self,
        host="localhost",
        database="psql_db",
        user="psql_user",
        password="root",
        port="5432",
        echo=False,
    ) -> None:
        
        dialect = 'postgresql'
        driver = 'psycopg2'
        url = f'{dialect}+{driver}://{user}:{password}@{host}:{port}/{database}'
        if url not in self.__shared_engines:
            self.__shared_engines[url] = create_engine(url, echo=echo)
        
        self.engine: Engine = self.__shared_engines[url]

    def select_model_by_id(self, id: str) -> Model:
        with Session(self.engine) as session:
            stmt = select(Model).where(Model.id == id)
            picture = session.scalars(stmt).one_or_none()
        return picture
    
    # TODO: add update method

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)

if __name__ == '__main__':
    handler = PostgreSQLHandler()
    handler.create_tables()