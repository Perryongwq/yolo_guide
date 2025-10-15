from amqp_logger2 import amqp_logger2
import logging,json
from datetime import datetime
from functools import lru_cache
import requests,gzip,io,sys,os,json
import pandas as pd

logger = logging.getLogger(__name__)
format = "%Y/%m/%d %H:%M:%S"

class utilfunc:
    ApplicationID = ''
    func_name = ''
    
    def __init__(self,applicationid,func_name):
        self.ApplicationID = applicationid
        self.func_name = func_name
        print(f'util func class initialize {datetime.now()}')
        
    def __del__(self):
        print(f'util func class destroyed {datetime.now()}')
   
    def amqp_message(self,errmsg):
        exc_type, exc_obj, exc_tb = sys.exc_info()
        
        if exc_tb is not None:
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            line_no = exc_tb.tb_lineno
        else:
            fname = "None"
            line_no = "None"
        
        cdatetime = datetime.now().strftime(format)
        errmsg = (f'{cdatetime} : {exc_type} : {fname} : Line {line_no} : {errmsg}')
        MQmessage = json.dumps({'programs': self.ApplicationID, 'functions' : self.func_name, 'errors': errmsg})   

        return MQmessage

    def app_start(self): 
        # MQmessage = self.amqp_message('Program Start')
        amqplogger = amqp_logger2(self.ApplicationID)
        amqplogger.send_events(self.amqp_message('Program Start'))

    def app_exit(self): 
        # MQmessage = self.amqp_message('Program exit gracefully')
        amqplogger = amqp_logger2(self.ApplicationID)        
        amqplogger.send_events(self.amqp_message('Program exit gracefully'))    

    def send_critical(self, errmsg): 
        amqplogger = amqp_logger2(self.ApplicationID)        
        amqplogger.send_critical(self.amqp_message(errmsg)) 
                        
    def get_remote_dfgz(self,url):
        headers = {
        'Accept-Encoding': 'gzip'
        }        
        df = pd.DataFrame()
                    
        response = requests.get(url, stream=True,headers=headers)
        if response.status_code == 200:
            with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
                df = pd.read_parquet(gz)
            # print(df)

        return df

    def get_remote_df(self,url):
        headers = {
        'Accept-Encoding': 'gzip'
        }    
       
        df = pd.DataFrame()
        try :
                       
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            
            # print(f'remoteurl : {url} : {response.status_code}')         
            if response.status_code == 200:
                parquet_buffer = io.BytesIO(response.content)
                df = pd.read_parquet(parquet_buffer)
        except requests.RequestException as e:
            self.send_critical(e)
                 
        return df   
    
    def get_remote_df_apikey(self,url,apikey):
        headers = {
        "x-api-key" :apikey
        }    
       
        df = pd.DataFrame()
        try :
                       
            response = requests.get(url,headers=headers)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            
            # print(f'remoteurl : {url} : {response.status_code}')         
            if response.status_code == 200:
                parquet_buffer = io.BytesIO(response.content)
                df = pd.read_parquet(parquet_buffer)
        except requests.RequestException as e:
            self.send_critical(e)
            print( e)
                 
        return df  

    def get_remote_df_post(self,url,data, fastapikey):
        headers = {
        'Accept-Encoding': 'gzip'
        }    

        df = pd.DataFrame()
        try :
                       
            response = requests.post(url, data=data, headers={"Content-Type":"application/json", "x-api-key" : fastapikey})
            response.raise_for_status()  # Raises an HTTPError for bad responses
                   
            if response.status_code == 200:
                parquet_buffer = io.BytesIO(response.content)
                df = pd.read_parquet(parquet_buffer)
        except requests.RequestException as e:
            self.send_critical(e)
                 
        return df    
            
