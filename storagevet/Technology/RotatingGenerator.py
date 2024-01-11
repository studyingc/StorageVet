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
Rotating Generator Class

CT: (Combustion Turbine) or gas turbine
ICE (Internal Combustion Engine)
DieselGenset or diesel engine-generator that is a single unit
    - can independently supply electricity allowing them to serve backup power
CHP (Combined Heat and Power)
    - also includes heat recovery

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""
import cvxpy as cvx
import numpy as np
import pandas as pd
from storagevet.Technology.DistributedEnergyResource import DER
from storagevet.ErrorHandling import *


class RotatingGenerator(DER):
    """ A Rotating Generator Technology

    """

    def __init__(self, params):
    # 해당 클래스의 인스턴스를 초기화하는 데 사용
        """ Initialize all technology with the following attributes.

        Args:
            params (dict): Dict of parameters for initialization
        """

        TellUser.debug(f"Initializing {__name__}")
        # create generic technology object
        super().__init__(params)
        # 부모 클래스인 DER 클래스의 생성자를 호출
        # input params  UNITS ARE COMMENTED TO THE RIGHT
        self.technology_type = 'Generator'
        self.rated_power = params['rated_capacity']  # kW/generator
        self.p_min = params['min_power']  # kW/generator
        self.variable_om = params['variable_om_cost']  # $/kWh
        self.fixed_om = params['fixed_om_cost']  # $/yr

        self.capital_cost_function = [params['ccost'],  # $/generator
                                      params['ccost_kW']]

        self.n = params['n']  # generators
        self.fuel_type = params['fuel_type']

        self.is_electric = True
        self.is_fuel = True
        # 클래스의 속성들을 초기화
        # 클래스의 속성들은 params 딕셔너리에서 가져온 값들을 사용하여 초기화

    def get_capex(self, **kwargs):
    # 기술의 자본비용(capex)을 계산하는 데 사용
        """ Returns the capex of a given technology
                """
        # self.capital_cost_function은 [cost_per_generator, cost_per_kW] 값을 가지게 됨
        # [self.n, self.discharge_capacity()]: 이 부분은 np.dot 함수를 사용하여 두 벡터의 내적(dot product)을 계산
        return np.dot(self.capital_cost_function, [self.n, self.discharge_capacity()])
        # 생성기 수와 방전 용량에 따른 자본비용을 내적한 값을 반환

    def discharge_capacity(self):
    # 최대 방전량을 반환
        """

        Returns: the maximum discharge that can be attained

        """
        # self.rated_power는 하나의 발전기의 등급이고, self.n은 발전기의 수입
        return self.rated_power * self.n
        # self.rated_power * self.n은 전체 발전기 수에 따른 최대 방전량을 나타냄
        # get_capex 메서드에서 사용 

    def qualifying_capacity(self, event_length):
    # RA(신뢰성 보조) 또는 DR(수요 응답) 이벤트에 참여하기 위해 DER가 방전할 수 있는 전력량
    # 시스템의 자격 부여 의무를 결정하는 데 사용
        """ Describes how much power the DER can discharge to qualify for RA or DR. Used to determine
        the system's qualifying commitment.

        Args:
            event_length (int): the length of the RA or DR event, this is the
                total hours that a DER is expected to discharge for

        Returns: int/float

        """
        return self.discharge_capacity()
        # self.discharge_capacity()를 호출하여 DER가 방전할 수 있는 최대 용량을 반환

    def initialize_variables(self, size):
    # 최적화 변수를 딕셔너리에 추가
    # 다양한 최적화 변수를 추가하고 각 변수에 대한 초기 값을 설정
        """ Adds optimization variables to dictionary

        Variables added:
            elec (Variable): A cvxpy variable equivalent to dis in batteries/CAES
                in terms of ability to provide services
            on (Variable): A cvxpy boolean variable for [...]

        Args:
            size (Int): Length of optimization variables to create

        """

        self.variables_dict = {'elec': cvx.Variable(shape=size, name=f'{self.name}-elecP', nonneg=True),
                               'udis': cvx.Variable(shape=size, name=f'{self.name}-udis', nonneg=True),
                               'on': cvx.Variable(shape=size, boolean=True, name=f'{self.name}-on')}
        # 세 가지 변수(elec, udis, on)를 생성하고 각 변수에 대한 초기 값을 설정

    def get_discharge(self, mask):
    # 해당 DER의 효과적인 방전을 나타냄 (elec 변수 반환)
        """ The effective discharge of this DER
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the discharge as a function of time for the

        """
        return self.variables_dict['elec']
        # 최적화 문제에서 DER의 방전을 나타냄

    def get_discharge_up_schedule(self, mask):
    # 해당 DER가 예약할 수 있는 상향 방향(그리드로 전력 공급)의 방전 전력 양을 나타냄
    # 얼마나 그리드로 전력을 공급할 수 있는지
        """ the amount of discharge power in the up direction (supplying power up into the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.rated_power * self.n - self.variables_dict['elec'] - self.p_min * (1 - self.variables_dict['on'])
        # self.rated_power * self.n는 DER의 최대 방전 용량
        # self.variables_dict['elec']는 DER가 실제로 방전하는 전력
        # self.p_min * (1 - self.variables_dict['on'])는 DER가 켜져 있지 않을 때의 최소 출력

    def get_discharge_down_schedule(self, mask):
    # 해당 DER가 예약할 수 있는 하향 방향(그리드에서 전력을 가져오는 방향)의 방전 전력 양을 나타냄
    # 해당 DER가 얼마나 그리드에서 전력을 가져올 수 있는지
        """ the amount of discharging power in the up direction (pulling power down from the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.variables_dict['elec'] - self.p_min * self.variables_dict['on']
        # self.variables_dict['elec']는 DER가 실제로 방전하는 전력
        # self.p_min * self.variables_dict['on']는 DER가 켜져 있을 때의 최소 출력을 나타냄

    def get_uenergy_decrease(self, mask):
    # 해당 DER가 분배 그리드에서 가져오는 에너지 양을 나타냄
        """ the amount of energy in a timestep that is taken from the distribution grid

        Returns: the energy throughput in kWh for this technology

        """
        return self.dt * self.variables_dict['udis']
        # self.dt는 각 타임 스텝의 길이(시간 간격)를 나타냄
        # self.variables_dict['udis']는 해당 DER가 실제로 분배 그리드에서 가져오는 방전 전력을 나타냄
        # DER의 에너지 감소량(energy throughput)을 계산

    def objective_function(self, mask, annuity_scalar=1):
    # 해당 DER의 목적 함수를 생성
    # 해당 기술의 고정 및 가변 운영 및 연료 비용으로 구성
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            mask (Series): Series of booleans used, the same length as case.power_kW
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                        the entire project lifetime (only to be set iff sizing)

        Returns:
            self.costs (Dict): Dict of objective costs
        """
        total_out = self.variables_dict['elec'] + self.variables_dict['udis']
        # self.variables_dict['elec']는 DER가 실제로 방전하는 전력,  self.variables_dict['udis']는 해당 DER가 실제로 분배 그리드에서 가져오는 방전 전력
        costs = {
            self.name + ' fixed': self.fixed_om * annuity_scalar, # self.fixed_om은 고정 운영 및 유지비
            self.name + ' variable': cvx.sum(self.variable_om * self.dt * annuity_scalar * total_out), # self.variable_om은 가변 운영 및 유지비
            # fuel cost in $/kW
            self.name + ' fuel_cost': cvx.sum(total_out * self.heat_rate * self.fuel_cost * self.dt * annuity_scalar) # self.heat_rate은 열량 효율
            # annuity_scalar는 해당 비용이 프로젝트 전체 수명 동안 어떻게 가중되는지를 제어하는 스칼라 값
        }
        return costs

    def constraints(self, mask):
    # 해당 DER의 제약 조건 목록을 생성
        """ Builds the master constraint list for the subset of timeseries data being optimized.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """
        constraint_list = []
        elec = self.variables_dict['elec']
        on = self.variables_dict['on']

        constraint_list += [cvx.NonPos((on * self.p_min) - elec)]
        constraint_list += [cvx.NonPos(elec - (on * self.rated_power * self.n))]
        # on은 기술이 활성화되어 있는지 여부를 나타내는 이진 변수
        # 방전이 최소 및 최대 값 사이에 있어야 한다는 것을 나타냅

        return constraint_list

    def timeseries_report(self):
    # DER의 최적화 결과 요약
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that
            summarize the results pertaining to this instance

        """
        tech_id = self.unique_tech_id()
        results = pd.DataFrame(index=self.variables_df.index)
        solve_dispatch_opt = self.variables_df.get('elec')
        # 최적화 결과 데이터프레임에서 'elec' 열을 가져옴
        if solve_dispatch_opt is not None:  #'elec' 열이 존재하는 경우
            results[tech_id + ' Electric Generation (kW)'] = \
                self.variables_df['elec']
                #  'elec' 열을 결과 데이터프레임에 'Electric Generation (kW)' 열로 추가
            results[tech_id + ' On (y/n)'] = self.variables_df['on']
            # 'on' 열을 결과 데이터프레임에 'On (y/n)' 열로 추가
            results[tech_id + ' Energy Option (kWh)'] = \
                self.variables_df['udis'] * self.dt
            # 'udis' 열에 self.dt를 곱하여 'Energy Option (kWh)' 열로 추가
        return results

    def proforma_report(self, apply_inflation_rate_func, fill_forward_func, results):
    # DER(Value Stream)가 참여한 결과에 대한 proforma(투자 수익/비용 계산서)를 생성
    # 각 연도별로 고정 및 가변 운영 및 유지보수 비용, 누적 에너지 디스패치량을 계산하여 결과로 반환
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            apply_inflation_rate_func:
            fill_forward_func:
            results (pd.DataFrame):

        Returns: A DateFrame of with each year in opt_year as the index and
            the corresponding value this stream provided.

        """
        pro_forma = super().proforma_report(apply_inflation_rate_func, fill_forward_func, results)
        # 상위 클래스(DER)에서 상속한 proforma_report 메서드를 호출하여 초기화된 proforma를 얻음
        tech_id = self.unique_tech_id()
        if self.variables_df.index.empty:
            return pro_forma
        optimization_years = self.variables_df.index.year.unique()
        # 최적화 결과의 연도를 추출

        # OM COSTS
        om_costs = pd.DataFrame()
        # 고정 및 가변 O&M 비용을 저장할 빈 데이터프레임을 생성
        cumulative_energy_dispatch_kw = pd.DataFrame()
        # 누적 에너지 디스패치량을 저장할 빈 데이터프레임을 생성
        elec = self.variables_df['elec']
        udis = self.variables_df['udis']
        dis_column_name = tech_id + ' Cumulative Energy Dispatch (kW)'
        variable_column_name = tech_id + ' Variable O&M Costs'
        # 결과 데이터프레임의 열 이름을 정의
        for year in optimization_years: # 각 연도에 대한 루프를 실행
            index_yr = pd.Period(year=year, freq='y')
            # 현재 연도에 해당하는 Period 객체를 생성
            # add fixed o&m costs
            om_costs.loc[index_yr, self.fixed_column_name()] = -self.fixed_om
            # 현재 연도에 대한 고정 O&M 비용을 빈 데이터프레임에 추
            # add variable costs
            elec_sub = elec.loc[elec.index.year == year]
            udis_sub = udis.loc[udis.index.year == year]
            # 현재 연도에 해당하는 전기 세대 및 에너지 디스패치 데이터를 추출
            om_costs.loc[index_yr, variable_column_name] = -self.variable_om
            # 현재 연도에 대한 가변 O&M 비용을 빈 데이터프레임에 추가
            cumulative_energy_dispatch_kw.loc[index_yr, dis_column_name] = np.sum(elec_sub) + np.sum(udis_sub)
            # 현재 연도에 대한 누적 에너지 디스패치량을 빈 데이터프레임에 추가

        # 프로포르마에 연료 및 O&M 비용을 추가하는 부분
        # fill forward (escalate rates)
        om_costs = fill_forward_func(om_costs, None, is_om_cost = True)
        # 비용을 전달하고 있음을 나타내는 플래그로 is_om_cost가 True로 설정

        # interpolate cumulative energy dispatch between analysis years
        #   be careful to not include years labeled as Strings (CAPEX)
        years_list = list(filter(lambda x: not(type(x) is str), om_costs.index))
        # 문자열로 레이블이 지정된 연도를 제외하고 연도 목록을 추출
        analysis_start_year = min(years_list).year
        analysis_end_year = max(years_list).year
        # 분석의 시작 및 종료 연도를 결정
        cumulative_energy_dispatch_kw = self.interpolate_energy_dispatch(
            cumulative_energy_dispatch_kw, analysis_start_year, analysis_end_year, None)
        # 누적에너지 디스패치 데이터를 분석 연도 사이에서 보간하는 메서드를 호출
        # calculate om costs in dollars, as rate * energy
        # fixed om is already in $
        # variable om
        om_costs.loc[:, variable_column_name] = om_costs.loc[:, variable_column_name] * self.dt * cumulative_energy_dispatch_kw.loc[:, dis_column_name]
        # 가변 O&M 비용을 계산하고, 누적 에너지 디스패치량과 시간 간격을 곱하여 에너지로 변환
        # append with super class's proforma
        pro_forma = pd.concat([pro_forma, om_costs], axis=1)
        # 고정 및 가변 O&M 비용을 포함하여 proforma를 업데이트
      
        # fuel costs in $/kW
        fuel_costs = pd.DataFrame()
        # 연료 비용을 저장할 빈 데이터프레임을 생성
        fuel_col_name = tech_id + ' Fuel Costs'
        # 연료 비용 열의 이름을 정의
        for year in optimization_years: # 연도별로 루프를 돌며
            elec_sub = elec.loc[elec.index.year == year]
            udis_sub = udis.loc[udis.index.year == year]
            # add fuel costs in $/kW
            fuel_costs.loc[pd.Period(year=year, freq='y'), fuel_col_name] = -np.sum(self.heat_rate * self.fuel_cost * self.dt * (elec_sub + udis_sub))
            # 전기 세대 및 에너지 디스패치 데이터를 사용하여 연료 비용을 계산하고 데이터프레임에 추가
        # fill forward
        fuel_costs = fill_forward_func(fuel_costs, None)
        # 연료 비용에 대해 fill forward 메서드를 적용
        # append with super class's proforma
        pro_forma = pd.concat([pro_forma, fuel_costs], axis=1)
        # 연료 비용을 포함하여 proforma를 최종적으로 업데이트하고 반환

        return pro_forma
