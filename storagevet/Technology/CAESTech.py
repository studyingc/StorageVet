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
CAESTech.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""
from storagevet.Technology.EnergyStorage import EnergyStorage
import cvxpy as cvx
import pandas as pd
import numpy as np
import storagevet.Library as Lib
from storagevet.ErrorHandling import *


class CAES(EnergyStorage):
    """ CAES class that inherits from Storage.
        ESS는 전기화학적 에너지저장 뿐만 아니라 기계적으로 저장이 가능하며 이 중 한아가 압력 포텐션(CAES) 방식이다.
        CAES는 양수발전과 함께 대용량 장주기 저장장치이다.
        압축기를 통해 압축공기에너지로 변환한 뒤 별도의 공간에 저장하고 필요시 압축공기를 이용하여 발전한다.
        
    """

    def __init__(self, params): #코드는 CAES (압축 공기 에너지 저장) 클래스를 초기화하는데 사용되는 __init__ 메서드
        """ Initializes a CAES class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
            params (dict): params dictionary from dataframe for one case
        """

        TellUser.debug(f"Initializing {__name__}")
        # 디버그 메시지를 통해 초기화가 진행 중임을 알림
        self.tag = 'CAES'
        # 객체에 'CAES' 태그를 설정
        # create generic technology object
        super().__init__(params)
        # 모 클래스의 __init__ 메서드를 호출하여 일반적인 기술 객체를 만듬
        # add CAES specific attributes
        self.tag = 'CAES'
        self.fuel_type = params['fuel_type']
        # CAES의 연료 유형을 설정
        self.heat_rate = 1e-3 * params['heat_rate_high']   # MMBtu/MWh ---> MMBtu/kWh
        # 열량 비율을 설정합니다. 1e-3를 곱해 MMBtu/MWh에서 MMBtu/kWh로 변환
        self.is_fuel = True
        # is_fuel 속성을 True로 설정하여 CAES가 연료를 사용하는 기술임을 나타냄

    def initialize_degradation_module(self, opt_agg):
        """
        bypass degradation by doing nothing here
        """
        pass
        # 초기 변질 모듈을 초기화하는 데 사용되는 것으로 보이지만, 여기에서는 아무런 동작을 수행하지 않고 넘어가도록 설정

    def objective_function(self, mask, annuity_scalar=1): # 주어진 제어 마스크에 대한 연료 비용 및 O&M 비용 목적 함수을 생성
        """ Generates the objective costs for fuel cost and O&M cost

         Args:
            mask (Series): Series of booleans used, the same length as case.power_kw
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing)

        Returns:
            costs (Dict): Dict of objective costs

        """
        # get generic Tech objective costs
        costs = super().objective_function(mask, annuity_scalar)
        # 부모 클래스의 objective_function 메서드를 호출하여 일반적인 기술의 목적 함수 비용
        total_out = self.variables_dict['dis'] + self.variables_dict['udis']
        # 출력 변수에서 방전 및 방전 상태의 합을 계산
        # add fuel cost expression in $/kWh
        fuel_exp = cvx.sum(total_out * self.heat_rate * self.fuel_cost * self.dt * annuity_scalar)
        # 연료 비용을 계산하는 표현식을 작성합니다. 출력, 열량 비율, 연료 비용, 시간 간격 및 연간 스칼라를 사용하여 계산
        costs.update({self.name + ' fuel_cost': fuel_exp})
        # costs 딕셔너리에 새로 계산된 연료 비용을 추가
        
        return costs
        # 업데이트된 목적 함수 비용 딕셔너리를 반환

    def proforma_report(self, apply_inflation_rate_func, fill_forward_func, results):
        # 부모 클래스의 proforma_report 메서드를 호출한 후에 CAES에 특화된 연료 비용을 추가하는 작업을 수행
        pro_forma = super().proforma_report(apply_inflation_rate_func, fill_forward_func, results)
        # 부모 클래스의 proforma_report 메서드를 호출하여 일반 기술의 pro forma 보고서를 얻음
        if self.variables_df.index.empty:
        # 변수 데이터 프레임이 비어 있는 경우
            return pro_forma
            # 부모 클래스의 pro forma를 그대로 반환
        tech_id = self.unique_tech_id()
        # 기술 ID를 가져옴
        optimization_years = self.variables_df.index.year.unique()
        # 최적화 연도 목록을 가져옴
        dis = self.variables_df['dis']
        # 방전 및 방전 상태 데이터를 가져옴
        udis = self.variables_df['udis']
        # 방전 및 방전 상태 데이터를 가져옴

        # add CAES fuel costs in $/kW
        fuel_costs = pd.DataFrame()
        # 연료 비용을 저장할 데이터 프레임을 초기화
        fuel_col_name = tech_id + ' Fuel Costs'     
        for year in optimization_years:
        # 각 최적화 연도에 대해 연료 비용을 계산
            dis_sub = dis.loc[dis.index.year == year]
            udis_sub = udis.loc[udis.index.year == year]
            # 해당 연도의 방전 및 방전 상태 데이터를 가져옴       
            # add fuel costs in $/kW
            fuel_costs.loc[pd.Period(year=year, freq='y'), fuel_col_name] = -np.sum(self.heat_rate * self.fuel_cost * self.dt * (dis_sub + udis_sub))
            # 연료 비용을 계산하고 연료 비용 데이터 프레임에 추가
        # fill forward
        fuel_costs = fill_forward_func(fuel_costs, None)
        # fill forward 함수를 사용하여 누락된 값들을 앞쪽으로 채움
        # append with super class's proforma
        pro_forma = pd.concat([pro_forma, fuel_costs], axis=1)
        # CAES의 연료 비용을 포함한 데이터 프레임을 부모 클래스의 pro forma에 추가

        return pro_forma
        # 업데이트된 pro forma 데이터 프레임을 반환
