import enum

class REQ_TYPE(enum.Enum):
    Prefill_DECODE = enum.auto()
    AGENT_PREFILL = enum.auto()


class AGENT_PREFILL_STATUS(enum.Enum):
    AGENT_PREILL_ING = enum.auto()
    AGENT_PREFILL_DONE = enum.auto()

class WaitingReq:
    def __init__(self, rid, prompt, output_text):
        self.rid = rid 
        self.prompt = prompt
        self.output_text = output_text

class AgentAData:
    def __init__(self, request_id:str, output_text, output_text_len):
        self.request_id = request_id
        self.output_text = output_text
        self.finished = False
        self.output_text_len = output_text_len
        #self.req_type = REQ_TYPE.Prefill_DECODE

class AgentBData:
    def __init__(self, request_id:str,output_text, output_text_len):
        self.request_id = request_id
        self.output_text = output_text
        self.finished = False
        self.output_text_len = output_text_len
        #self.req_type = REQ_TYPE.Prefill_DECODE

class ReactReq:
    def __init__(self, rid, arr,prompt):
        self.arr = arr
        self.rid = rid 
        self.total_duration = 0
        self.total_token = 0 
        self.send_time = 0  
        self.finished = False
        self.prompt = prompt
        self.req_type = REQ_TYPE.Prefill_DECODE


class BSendData:
    def  __init__(self, request_id, prompt, output_text_len, output_text):
        self.request_id = request_id
        self.prompt = prompt
        self.output_text_len = output_text_len
        self.output_text = output_text