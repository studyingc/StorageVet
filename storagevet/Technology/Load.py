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
Load

This Python class contains methods and attributes specific for technology analysis within DERVET.
"""

__author__ = 'Halley Nathwani'
__credits__ = ['Miles Evans', 'Andres Cortes', 'Evan Giarta', 'Halley Nathwani', 'Micah Botkin-Levy', 'Yekta Yazar']
__license__ = 'EPRI'
__maintainer__ = ['Halley Nathwani', 'Evan Giarta', 'Miles Evans']
__email__ = ['hnathwani@epri.com', 'egiarta@epri.com', 'mevans@epri.com']
__version__ = '2.1.0.2'

import numpy as np
import pandas as pd
from storagevet.Technology.DistributedEnergyResource import DER
import storagevet.Library as Lib
import cvxpy as cvx
from storagevet.ErrorHandling import *


class Load(DER):
    """ An Site Load object

    """

    def __init__(self, params):
    # 어떤 로드(부하)를 나타내는지에 대한 속성들을 설정
        """ Initialize all technology with the following attributes.

        Args:
            params (dict): Dict of parameters for initialization
        """
        TellUser.debug(f"Initializing {__name__}")
        self.tag = 'Load'
        # create generic technology object
        super().__init__(params)
        self.technology_type = 'Load'
        self.tag = 'Load'
        self.dt = params['dt']
        self.value = params['site_load']
        # 변수 설정

    def zero_column_name(self):
    # 로드(부하)에 대해 0으로 설정된 자본 비용이 없다는 것
        return None  # Loads do not have capital costs
        # 로드는 전력 네트워크에서 에너지를 소비하는 역할을 하므로 대부분의 경우 자본 비용이 필요하지 않음

    def grow_drop_data(self, years, frequency, load_growth):
    # 부하의 데이터를 성장 또는 삭제하는데 사용
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.
        # 추가로 증가된 데이터 이후, 최적화 이전 
        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation

        """
        self.value = Lib.fill_extra_data(self.value, years, load_growth, frequency)
        # 지정된 연도 및 빈도에 따라 로드 데이터를 성장
        self.value = Lib.drop_extra_data(self.value, years)
        # 지정된 연도 이외의 추가 데이터를 삭제

        # grow_drop_data 메서드를 호출하여 로드 데이터를 시뮬레이션에 맞게 설정한 후에는 
        # 최적화를 실행하기 전에 이 메서드를 호출하여 데이터를 준비

    def get_charge(self, mask):
        # 시뮬레이션에서 로드 데이터를 가져와서 해당 데이터를 반환
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the charge as a function of time for the

        """
        return cvx.Parameter(value=self.value[mask].values, shape=sum(mask), name='SiteLoad')
        # value 속성은 mask에 해당하는 시간대의 데이터로 설정되어 있고, shape는 해당 데이터의 크기로 설정
        # 로드 데이터 최적화

    def effective_load(self):
    # 마이크로그리드나 접속 지점에서 볼 수 있는 효과적인 로드를 반환
        """ Returns the load that is seen by the microgrid or point of interconnection

        """
        return self.value.loc[:]
        # self.value에 저장된 전체 로드 데이터를 반환

    def timeseries_report(self):
    # 최적화 결과를 요약하여 사용자 친화적인 열 헤더를 포함한 시계열 데이터프레임을 반환
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that
            summarize the results pertaining to this instance

        """
        results = pd.DataFrame(index=self.variables_df.index)
        results[f"{self.unique_tech_id()} Original Load (kW)"] = \
            self.value.loc[:]
        # Original Load (kW)라는 열을 추가하고 해당 열에는 기존 로드 데이터인 self.value를 저장
        return results

    def drill_down_reports(self, monthly_data=None, time_series_data=None, technology_summary=None, sizing_df=None):
    # 서비스와 관련된 데이터프레임을 계산하고 사용자에게 보고하는 역할
        """Calculates any service related dataframe that is reported to the user.

        Args:
            monthly_data:
            time_series_data:
            technology_summary:
            sizing_df:

        Returns: dictionary of DataFrames of any reports that are value stream specific
            keys are the file name that the df will be saved with
        """
        # DESIGN PLOT (peak load day)
        max_day = time_series_data['Total Load (kW)'].idxmax().date()
        max_day_data = time_series_data[time_series_data.index.date == max_day]
        time_step = pd.Index(np.arange(0, 24, self.dt), name='Timestep Beginning')
        peak_day_load = pd.DataFrame({'Date': max_day_data.index.date,
                                      'Load (kW)': max_day_data['Total Load (kW)'].values,
                                      'Net Load (kW)': max_day_data['Net Load (kW)'].values}, index=time_step)
                                      # Date: 최대 부하가 발생한 날짜
                                      # Load: 해당 날짜의 시간별 총 부하
                                      # Net Load: 해당 날짜의 시간별 순 부하
        return {'peak_day_load': peak_day_load}
        # "peak_day_load"라는 키로 peak_day_load 데이터프레임을 반환
        # 데이터프레임은 최대 부하가 발생한 날의 시간별로 부하 및 순 부하를 나타내는 것
