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
VoltVar.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
THIS CLASS HAS NOT BEEN VALIDATED OR TESTED TO SEE IF IT WOULD SOLVE.
"""

from storagevet.ValueStreams.ValueStream import ValueStream
import math
import pandas as pd
import logging
from storagevet.SystemRequirement import Requirement
import storagevet.Library as Lib


class VoltVar(ValueStream):
    """ VoltVar 클래스: 반응 전력 지원, 전압 제어, 전력 품질을 다루는 클래스입니다.
        각 서비스는 PreDispService 클래스의 하위 클래스로 구현될 것입니다.
    """

    def __init__(self, params):
        """ 목적 함수를 생성하고 제약 조건을 찾아 생성합니다.
          Args:
            params (Dict): 입력 매개변수
        """

        # generate the generic service object
        ValueStream.__init__(self, 'Volt Var', params)

        # add voltage support specific attributes
        self.vars_percent = params['percent'] / 100
        self.price = params['price']

        self.vars_reservation = 0

    def grow_drop_data(self, years, frequency, load_growth):
        """  데이터를 성장시키거나 추가로 들어온 데이터를 제거합니다. 최적화를 실행하기 전에 add_growth_data 메서드를 호출한 후에 이 메서드들이 호출되어야 합니다.

        Args:
            years (List): 분석이 수행될 연도 목록
            frequency (str): 시계열 데이터의 주기
            load_growth (float): 이 시뮬레이션의 부하 성장률의 백분율 또는 소수값

        """
        # 추가된 데이터를 성장시킵니다.
        self.vars_percent = Lib.fill_extra_data(self.vars_percent, years, 0, 'M')
        # 추가로 들어온 데이터를 제거합니다.
        self.vars_percent = Lib.drop_extra_data(self.vars_percent, years)

    def calculate_system_requirements(self, der_lst):
        """ 다른 Value Stream이 활성화되어 있더라도 충족해야 할 시스템 요구 사항을 계산합니다. 그러나 이러한 요구 사항은 분석에 활성화된 기술에 따라 달라집니다.

        Args:
            der_lst (list): 시나리오에서 초기화된 DER(분산 에너지 자원) 목록

        """
        # PV가 포함되어 있고 'dc'로 연결되어 있는지 확인합니다. TODO: 이 부분을 수정하여 서비스가 작동하도록 합니다.
        pv_max = 0
        inv_max = 0
        # if 'PV' in der_dict.keys:
        #     if der_dict['PV'].loc == 'dc':
        #         # use inv_max of the inverter shared by pv and ess and save pv generation
        #         inv_max = der_dict['PV'].inv_max
        #         pv_max = der_dict['PV'].generation
        # else:
        #     # otherwise just use the storage's rated discharge
        #     inv_max = der_dict['Storage'].dis_max_rated

        # # save load
        # self.load = load_data['load']

        self.vars_reservation = self.vars_percent * inv_max

        # constrain power s.t. enough vars are being outted as well
        power_sqrd = (inv_max**2) - (self.vars_reservation**2)

        dis_max = math.sqrt(power_sqrd) - pv_max
        ch_max = math.sqrt(power_sqrd)

        dis_min = 0
        ch_min = 0

        self.system_requirements = {
            'ch_max': ch_max,
            'dis_min': dis_min,
            'ch_min': ch_min,
            'dis_max': dis_max}

    def proforma_report(self, opt_years, apply_inflation_rate_func, fill_forward_func, results):
        """ Value Stream에 해당하는 proforma를 계산합니다.
        Args:
             opt_years (list): 최적화 문제가 실행된 연도 목록
             apply_inflation_rate_func: 인플레이션 비율 함수 적용
             fill_forward_func:
             results (pd.DataFrame): 모든 최적화 변수 솔루션을 포함한 DataFrame


        Returns: 각 연도를 인덱스로 사용하고 이 가치 스트림이 제공한 해당 값을 포함하는 DataFrame의 튜플

        """
        # 상위 클래스인 ValueStream의 proforma_report 메서드를 호출하여 초기 proforma를 계산합니다.
        proforma = ValueStream.proforma_report(self, opt_years, apply_inflation_rate_func,
                                                     fill_forward_func, results)
        # proforma DataFrame의 열 이름을 수정하여 현재 서비스의 이름과 'Value'를 추가합니다.
        proforma.columns = [self.name + ' Value']

        # 각 연도에 대해 self.price 값을 proforma에 추가합니다.
        for year in opt_years:
            proforma.loc[pd.Period(year=year, freq='y')] = self.price
        # 인플레이션 비율 적용
        proforma = apply_inflation_rate_func(proforma, None, min(opt_years))

        return proforma

    def update_yearly_value(self, new_value: float):
        """ 이 서비스의 연간 가치에 연결된 속성을 업데이트합니다. (CBA에서 사용됨)
        Args:
            new_value (float): 이 서비스를 제공하는 데 할당된 연간 달러 가치

        """
        self.price = new_value
