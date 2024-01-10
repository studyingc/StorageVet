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
CurtailPVPV.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""
from storagevet.Technology.DistributedEnergyResource import DER
import cvxpy as cvx
import pandas as pd
import storagevet.Library as Lib
from storagevet.ErrorHandling import *


class PV(DER):
    """ Assumes perfect foresight. Ability to curtail PV generation

    """

    def __init__(self, params):
    # PV 클래스를 초기화, 태양광 발전소에 대한 모델
        """ Initializes a PV class where perfect foresight of generation is assumed.
        It inherits from the technology class. Additionally, it sets the type and physical constraints of the
        technology.

        Args:
            params (dict): Dict of parameters
        """
        TellUser.debug(f"Initializing {__name__}")
        self.tag = 'PV'
        # create generic technology object
        super().__init__(params)
        self.technology_type = "Intermittent Resource"
        # "Intermittent Resource"로 설정되어 있습니다. 이는 이 기술이 간헐적 자원임
        self.tag = 'PV'
        self.growth = params['growth']/100
        # 성장률을 나타냄
        self.curtail = True
        # 발전량이 그리드에 공급될 때 발전량의 일부가 낭비되어 전력을 전송하지 않는다는 것을 나타냄
        self.gen_per_rated = params['rated gen']
        # 발전소의 등급에 대한 매개변수
        self.rated_capacity = params['rated_capacity']
        # 발전소의 등급 용량
        self.loc = params['loc'].lower()
        # 발전소의 위치를 나타내는 매개변수
        self.grid_charge = params['grid_charge']
        # 그리드 충전 여부를 나타내는 매개변수
        self.inv_max = params['inv_max']
        # 인버터의 최대 용량을 나타내는 매개변수
        self.capital_cost_function = params['ccost_kW']
        # 자본 비용 함수를 나타내는 매개변수
        self.fixed_om = params['fixed_om_cost']  # $/yr
        #  고정 운영 및 유지 관리 비용을 나타내는 매개변수

    def grow_drop_data(self, years, frequency, load_growth):
    # 기술의 발전량(gen_per_rated)을 특정 성장률로 증가시키거나 불필요한 데이터를 제거하는 데 사용
    # 최적화를 실행하기 전에 add_growth_data 메서드 이후에 호출
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation

        """
        self.gen_per_rated = Lib.fill_extra_data(self.gen_per_rated, years, self.growth, frequency)
        # Lib.fill_extra_data 함수를 사용하여 발전량 데이터를 성장률에 따라 늘리고 불필요한 데이터를 추가
        self.gen_per_rated = Lib.drop_extra_data(self.gen_per_rated, years)
        # Lib.drop_extra_data 함수를 사용하여 불필요한 데이터를 제거
    # 통해 연도 및 성장률에 따라 발전량 데이터가 적절하게 갱신되고 최적화에 사용될 준비

    def get_capex(self, **kwargs):
    # DER(분산 에너지 자원)의 자본비용(CAPEX)을 최적화에 사용될 값으로 반환
        """

        Returns: the capex of this DER for optimization

        """
        return self.capital_cost_function * self.rated_capacity # (기술의 설치용량에 대한 자본비용 함수)*(기술의 등급 용량)
        # 설치용량에 대한 자본비용 함수를 등급 용량에 적용하여 최종 자본비용을 계산
        # 최적화 문제에서 기술의 자본 비용 고려

    def initialize_variables(self, size):
    # 최적화 변수를 딕셔너리에 추가, size: 최적화 변수의 길이
        """ Adds optimization variables to dictionary

        Variables added:
            pv_out (Variable): A cvxpy variable for the ac eq power outputted by the PV system

        Args:
            size (Int): Length of optimization variables to create

        """
        if self.curtail:
            self.variables_dict = {'pv_out': cvx.Variable(shape=size, name='pv_out', nonneg=True)}
            # shape=size는 최적화 변수의 크기를 지정하고, name='pv_out'은 변수의 이름을 설정, nonneg=True는 변수가 음수가 아닌 값

    def maximum_generation(self, label_selection=None, **kwargs):
    # PV 시스템의 최대 발전량을 계산하는 기능ㅇㅇㅇㅇㅇㅇㅇㅇ
        """ The most that the PV system could discharge.

        Args:
            label_selection: A single label, e.g. 5 or 'a',
                a list or array of labels, e.g. ['a', 'b', 'c'],
                a boolean array of the same length as the axis being sliced, e.g. [True, False, True]
                a callable function with one argument (the calling Series or DataFrame)

        Returns: valid array output for indexing (one of the above) of the max generation profile

        """
        if label_selection is None: # 만약 label_selection이 주어지지 않으면
            return self.gen_per_rated.values * self.rated_capacity
            # gen_per_rated 변수에 저장된 발전량 데이터에 rated_capacity를 곱하여 최대 발전량을 반환
        else:
            return self.gen_per_rated.loc[label_selection].values * self.rated_capacity
            # 선택한 라벨에 해당하는 발전량 데이터를 가져와 rated_capacity를 곱하여 반환

    def get_discharge(self, mask):
    # DER의 효과적인 방전량을 계산
        """ The effective discharge of this DER
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the discharge as a function of time for the

        """
        if self.curtail: # 만약 self.curtail이 True이면
            return self.variables_dict['pv_out']
            # variables_dict에서 'pv_out'에 해당하는 변수를 반환
        else:
            return cvx.Parameter(shape=sum(mask), name='pv_out', value=self.maximum_generation(mask))
            # cvx.Parameter를 사용하여 'pv_out'라는 이름의 파라미터를 만들고,
            # 해당 파라미터에 maximum_generation 메서드의 결과 값을 초기값으로 설정하여 반환
            # mask를 이용하여 해당 기간 동안의 최대 발전량을 사용

    def constraints(self, mask, **kwargs):
    # DER의 제약 조건을 만들어냄
     
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """
        constraint_list = []

        if self.loc == 'ac': # 만약 self.loc이 'ac'이면
            constraint_list += [cvx.NonPos(self.get_discharge(mask) - self.inv_max)]
            # self.get_discharge(mask)이 self.inv_max보다 크지 않도록 하는 제약조건이 포함
            constraint_list += [cvx.NonPos(- self.inv_max - self.get_discharge(mask))]
            # -self.inv_max - self.get_discharge(mask)가 0 이하이도록 하는 제약조건이 추가
        if self.curtail: # self.curtail이 True인 경우
            constraint_list += [cvx.NonPos(self.get_discharge(mask) - self.maximum_generation(mask))]
            # self.get_discharge(mask)이 self.maximum_generation(mask)보다 크지 않도록 하는 제약조건이 추가

        return constraint_list

    def timeseries_report(self):
    # DER의 최적화 결과를 요약하는 데이터프레임을 생성
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        tech_id = self.unique_tech_id()
        results = pd.DataFrame(index=self.gen_per_rated.index)
        # 결과를 저장할 빈 데이터프레임 results를 생성
        if self.curtail: # 만약 self.curtail이 True이면: 
            solve_dispatch_opt = self.variables_df.get('pv_out')
            if solve_dispatch_opt is not None:
                results[tech_id + ' Electric Generation (kW)'] = \
                    self.variables_df['pv_out']
            # self.variables_df에서 'pv_out' 열을 가져와서 'Electric Generation (kW)'이라는 열로 저장
        else:
            results[tech_id + ' Electric Generation (kW)'] = self.maximum_generation()
            # self.maximum_generation()을 호출하여 'Electric Generation (kW)' 열에 저장
        results[tech_id + ' Maximum (kW)'] = self.maximum_generation()
        # 'Maximum (kW)' 열에도 self.maximum_generation() 값을 저장
   
        return results

    def proforma_report(self, apply_inflation_rate_func, fill_forward_func, results):
    # 특정 가치 스트림에 참여하는 데 필요한 프로포르마(견적)를 계산
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            apply_inflation_rate_func:
            fill_forward_func:
            results (pd.DataFrame):

        Returns: A DateFrame of with each year in opt_year as the index and
            the corresponding value this stream provided.

        """
        pro_forma = super().proforma_report(apply_inflation_rate_func, fill_forward_func, results)
        # super().proforma_report()를 호출하여 상위 클래스에서 계산된 프로포르마를 가져옴
        if self.variables_df.index.empty:
            return pro_forma
            # self.variables_df.index가 비어 있으면 빈 프로포르마를 반환
        optimization_years = self.variables_df.index.year.unique()

        # OM COSTS
        om_costs = pd.DataFrame()
        for year in optimization_years: # 최적화 연도에 대해 반복하면서
            # add fixed o&m costs
            om_costs.loc[pd.Period(year=year, freq='y'), self.fixed_column_name()] = -self.fixed_om
            # 각 연도에 대한 고정 운영 및 유지 비용을 om_costs 데이터프레임에 추가
        # fill forward
        om_costs = fill_forward_func(om_costs, None)
        # fill_forward_func를 사용하여 누락된 값을 채웁
        # append will super class's proforma
        pro_forma = pd.concat([pro_forma, om_costs], axis=1)
        # 상위 클래스의 프로포르마와 om_costs를 합침
        return pro_forma
