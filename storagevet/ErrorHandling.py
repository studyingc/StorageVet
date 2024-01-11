"""
Copyright (c) 2023, Electric Power Research Institute

 All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
     * Neither the name of DER-VET nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
"""
This file contains all error classes. It also initialized and writes to the log
files to give the user feedback on what went wrong.
"""

import os
import logging
from pathlib import Path


class SolverError(Exception):                     """solver 관련 오류에 대한 일반적인 예외 클래스"""
    pass


class SolverInfeasibleError(Exception):           """solver의 해결이 불가능한 상태 오류"""
    pass


class SolverUnboundedError(Exception):            """solver의 해결이 무한한 상태 오류"""
    pass


class SystemRequirementsError(Exception):         """시스템 요구사항과 관련된 오류"""
    pass


class ModelParameterError(Exception):              """모델 매개변수와 관련된 오류"""
    pass


class TimeseriesDataError(Exception):             """시계열 데이터와 관련된 오류"""
    pass


class TimeseriesMissingError(Exception):          """시계열 데이터가 누락된 경우의 오류"""
    pass


class MonthlyDataError(Exception):                 """월별 데이터와 관련된 오류"""
    pass


class TariffError(Exception):                      """타리프와 관련된 오류"""
    pass


class ParameterError(ValueError):                   """기본 내장 ValueError 클래스를 상속받은 일반적인 매개변수 오류를 나타내는 사용자 정의 예외 클래스"""
    pass


class FilenameError(ValueError):                  """기본 내장 ValueError 클래스를 상속받은 파일 이름과 관련된 오류를 나타내는 사용자 정의 예외 클래스"""
    pass
 
class TellUser:
    @classmethod
    def create_log(cls, logs_path, verbose):                                       """logs_path (로그 디렉토리의 경로)와 verbose (메시지를 콘솔에 출력할지 여부를 나타내는 부울 값)를 매개변수로 사용"""
            os.makedirs(logs_path)
        except OSError:
            print("Creation of the logs_path directory %s failed. Possibly already created." % logs_path) if verbose else None
        else:
            print("Successfully created the logs_path directory %s " % logs_path) if verbose else None
        log_filename = logs_path / 'dervet_log.log'                                """dervet_log.log'라는 이름의 로그 파일을 지정된 로그 디렉토리 내에 만듬"""
        handler = logging.FileHandler(log_filename, mode='w')                      """로그를 파일로 저장하는 핸들러를 쓰기모드로 열도록 지정"""
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') """로그 메시지 형식 지정, (작성 시간/로그 레벨(DEBUG,ERROR 등)/실제 로그메시지"""
        handler.setFormatter(formatter)                                            """핸들러에 포매터 설정하며 파일 핸들러가 사용자가 지정한 형식으로 로그메시지를 기록하게 함"""
        cls.logger = logging.getLogger('Error')                                    """error의 로거를 반환하거나 새로 생성"""
        cls.logger.setLevel(logging.DEBUG)                                         """로거의 로깅수준을 설정/ DEBUG부터 ERROR까지 모든 메시지 기록"""
        cls.logger.addHandler(handler)                                             """로거에 핸들러 추가하여 이 로거로 전송된 로그 메시지가 파일에도 기록되도록 설정"""
        if verbose:
            # create console handler and set level to debug
            ch = logging.StreamHandler()                                           
            ch.setLevel(logging.DEBUG)                                             
            # add formatter to ch
            ch.setFormatter(formatter)
            # add ch to logger
            cls.logger.addHandler(ch)
        cls.logger.info('Started logging...')

    @classmethod
    def close_log(cls):
        for i in list(cls.logger.handlers):
            print(i)
            cls.logger.removeHandler(i)
            i.flush()                                     """내부 버퍼를 비워줌"""
            i.close()                                     

    @classmethod
    def debug(cls, msg):
        cls.logger.debug(msg)

    @classmethod
    def info(cls, msg):
        cls.logger.info(msg)

    @classmethod
    def warning(cls, msg):
        cls.logger.warning(msg)

    @classmethod
    def error(cls, msg):
        cls.logger.error(msg)
"""debug,ingo,warning,error 메서드를 사용하여 각 로그 레벨의 메시지를 기록"""
