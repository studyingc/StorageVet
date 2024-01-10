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
BatteryTech.py

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""
from .EnergyStorage import EnergyStorage
import numpy as np
import pandas as pd
import rainflow
from storagevet.ErrorHandling import *
from storagevet.Library import truncate_float, is_leap_yr
import cvxpy as cvx


class Battery(EnergyStorage):
    """ Battery class that inherits from Storage.

    """

    def __init__(self, params): # 전체적인 값의 초기화 함수 self와 params에 해당하는 값을 받아옴
     
        """ Initializes a battery class that inherits from the technology class.
        It sets the type and physical constraints of the technology.

        Args:
           params (dict): params dictionary from dataframe for one case
        """
     
        TellUser.debug(f"Initializing {__name__}")
        self.tag = 'Battery'
        # create generic storage object
     
        super().__init__(params)
        self.hp = params['hp']

        self.tag = 'Battery'
     
        # initialize degradation attributes
        self.cycle_life = params['cycle_life'] # cycle_life의 속성을 params 사전의 cycle_life 키의 값으로 초기화
        self.degrade_perc = 0 # 0으로 초기화
        self.soh_initial = 1 # 최초 배터리 SOC 1로 초기화#Initial SOC at the start of the project
        self.soh=1 # 배터리 SOC 1로 초기화#Initial SOC at the start of the project
        self.yearly_degrade = params['yearly_degrade'] / 100 #
        self.eol_condition = params['cycle_life_table_eol_condition'] / 100
        self.incl_cycle_degrade = bool(params['incl_cycle_degrade']) 
        self.degrade_data = None # degrade_data를 None으로 초기화
        self.counted_cycles = [] # 빈 리스트로 초기화

    def initialize_degradation_module(self, opt_agg):
        각종 속성 및 데이터 프레임을 초기화
        """
        Notes: Should be called once, after optimization levels are assigned, but before
        optimization loop gets called

        Args:
            opt_agg (DataFrame):

        Returns: None

        """
        """
        Dictionary: key를 통해 value를 얻는 자료형이다.
        Dataframe: 딕셔너리를 통해 각 칼럼에 대한 데이터를 저장한 후 딕셔너리를 DataFrame 클래스의 생선자 인자로 넘겨주면 DataFrame 객체가 생성된다.
        """
        if self.incl_cycle_degrade: #만약 incl_cycle_degrade의 값이 True 일때(= 사이클 감소를 고려할 떄)
            # initialize degradation dataframe
            # degrade_data 값은 none으로 초기화 돼 있음
            self.degrade_data = pd.DataFrame(index=['Optimization Start']+list(opt_agg.control.unique()))
            # degrade_data의 속성을 데이터 프레임을 만들어 optimization start와 opt_agg.control.unique()에서 가저온 고유한 값들을 인덱스로 사용
            self.degrade_data['degradation progress %'] = self.degrade_perc
            # 'degradation progress %' 열을 생성하고 self.degrade_perc 값을 할당
            self.degrade_data['state of health %'] = self.soh *100
            # 'state of health %' 열을 생성하고 self.soh에 100을 곱한 값을 할당 (self.soh = 1로 초기화 돼있음)
            self.degrade_data['effective energy capacity (kWh)'] = self.degraded_energy_capacity()
            # 'effective energy capacity (kWh)' 열을 생성하고 self.degraded_energy_capacity() 메서드의 반환값을 할당
            self.calc_degradation('Optimization Start', None, None)
            # calc_degradation 메서드를 호출하여 'Optimization Start' 시점에서의 감소를 계산합니다.
            # 이 메서드에 대한 구현은 코드 스니펫에서 제공되지 않았습니다.

 
    def degraded_energy_capacity(self):
        # 감소된 에너지 용량을 계산하여 반환
        """ Updates ene_max_rated and control constraints based on degradation percent
        Applies degrade percent to rated energy capacity

        TODO: use lookup table for energy cap to degredation percentage

        Returns:
            Degraded energy capacity
        """

        soh_change = self.degrade_perc
        # degrade_perc 속성에서 감소 비율을 가져와 soh_change 변수에 할당
        new_ene_max = max(self.ene_max_rated * (1 - soh_change), 0)
        # 감소된 에너지 최대값을 계산합니다. self.ene_max_rated는 초기에 설정된 최대 에너지 용량이고, 
        # soh_change를 감소 비율로 사용하여 현재의 최대 용량을 계산합니다. 
        # 최종적으로 max 함수를 사용하여 음수가 되지 않도록 합니다.
        return new_ene_max
        # 계산된 새로운 에너지 최대값을 반환

    def calc_degradation(self, opt_period, start_dttm, last_dttm):
        # 에너지 저장 장치의 감소율을 계산하고 업데이트하는 역할
        """ calculate degradation percent based on yearly degradation and cycle degradation

        Args:
            opt_period: the index of the optimization that occurred before calling this function, None if
                no optimization problem has been solved yet
            start_dttm (DateTime): Start timestamp to calculate degradation. ie. the first datetime in the optimization
                problem
            last_dttm (DateTime): End timestamp to calculate degradation. ie. the last datetime in the optimization
                problem

        A percent that represented the energy capacity degradation
        """

        # time difference between time stamps converted into years multiplied by yearly degrate rate
        if self.incl_cycle_degrade:
            cycle_degrade = 0
            yearly_degradation = 0
            #incl_cycle_degrade가 true라면 cycle_degrade와 yearly_degradation을 0으로 초기화
         
            if not isinstance(opt_period, str):
                # opt_period 문자열이 아닌 경우에만 실행
                # calculate degradation due to cycling iff energy values are given
                energy_series = self.variables_df.loc[start_dttm:last_dttm, 'ene']
                # 주어진 시간 범위(start_dttm부터 last_dttm까지)에서 'ene' 열의 에너지 값을 가져옴
                # Find the effective energy capacity
                eff_e_cap = self.degraded_energy_capacity()
                # 'degraded_energy_capacity' 메서드를 호출하여 감소된 에너지 용량을 계산

             
                #If using rainflow counting package uncomment following few lines
                # rainflow counting 알고리즘을 사용하여 주기별 사이클을 계산하는 부분
                # use rainflow counting algorithm to get cycle counts
                # cycle_counts = rainflow.count_cycles(energy_series, ndigits=4)
                #
                # aux_df = pd.DataFrame(cycle_counts, columns=['DoD', 'N_cycles'])
                # aux_df['Opt window'] = opt_period
                #
                # # sort cycle counts into user inputed cycle life bins
                # digitized_cycles = np.searchsorted(self.cycle_life['Cycle Depth Upper Limit'],[min(i[0]/eff_e_cap, 1) for i in cycle_counts], side='left')
      
                # use rainflow extract function to get information on each cycle
                cycle_extract=list(rainflow.extract_cycles(energy_series))
                # rainflow 알고리즘을 사용하여 주기별 사이클을 추출
                aux_df = pd.DataFrame(cycle_extract, columns=['rng', 'mean','count','i_start','i_end'])
                aux_df['Opt window'] = opt_period

                # sort cycle counts into user inputed cycle life bins
                digitized_cycles = np.searchsorted(self.cycle_life['Cycle Depth Upper Limit'],[min(i[0] / eff_e_cap, 1) for i in cycle_extract], side='left')
                # 주기별 사이클을 주어진 cycle depth upper limit에 따라 순서대로 정렬
                aux_df['Input_cycle_DoD_mapping'] = np.array(self.cycle_life['Cycle Depth Upper Limit'][digitized_cycles]*eff_e_cap)
                aux_df['Cycle Life Value'] = np.array(self.cycle_life['Cycle Life Value'][digitized_cycles] )

                self.counted_cycles.append(aux_df.copy())
                # 각 주기의 정보를 담고 잇는 데이터 프레임을 'counted_cycles'리스트에 추가
                # sum up number of cycles for all cycle counts in each bin
                cycle_sum = self.cycle_life.loc[:, :]
                # cycle_life 테이블을 복사하여 cycle_sum에 할당
                cycle_sum.loc[:, 'cycles'] = 0
                # cycle열을 0으로 초기화
                for i in range(len(cycle_extract)):
                    cycle_sum.loc[digitized_cycles[i], 'cycles'] += cycle_extract[i][2]
                # 각 주기에 대한 정보를 사용하여 cycle_sum에 주기별로 발생한 사이클 수를 누적
   
                # sum across bins to get total degrade percent
                # 1/cycle life value is degrade percent for each cycle
                cycle_degrade = np.dot(1/cycle_sum['Cycle Life Value'], cycle_sum.cycles)* (1 - self.eol_condition)
                # 각 주기에 대한 감소율을 계산하고, 이를 cycle_degrade에 누적

            if start_dttm is not None and last_dttm is not None: #start_dttm과 last_dttm이 존재할 경우
                # add the yearly degradation linearly to the # of years from START_DTTM to (END_DTTM + dt)
                days_in_year = 366 if is_leap_yr(start_dttm.year) else 365'
                # 윤년인지 확인해서 윤년이면 366일 그외는 365일로 설정
                portion_of_year = (last_dttm + pd.Timedelta(self.dt, unit='h') - start_dttm) / pd.Timedelta(days_in_year, unit='d')
                # 주어진 시간 간격 동안의 연간 감소를 계산
                yearly_degradation = self.yearly_degrade * portion_of_year

            # add the degradation due to time passing and cycling for total degradation
            degrade_percent = cycle_degrade + yearly_degradation
            # 연간 감소 및 주기적 감소를 합하여 총 감소율을 계산

            # record the degradation
            # the total degradation after optimization OPT_PERIOD must also take into account the
            # degradation that occurred before the battery was in operation (which we saved as SELF.DEGRADE_PERC)
            self.degrade_data.loc[opt_period, 'degradation progress %'] = degrade_percent + self.degrade_perc
            # 최적화 이후의 감소율을 degrade_data 데이터프레임에 기록
            self.degrade_perc += degrade_percent
            # 총 감소율을 누적하여 저장

            soh_new = self.soh_initial - self.degrade_perc
            # 총 감소율을 누적하여 저장
            self.soh = self.degrade_data.loc[opt_period, 'state of health %'] = soh_new
            # 에너지 저장 장치의 상태를 갱신, 최적화 이후의 상태를 degrade_data에 업데이트

            # apply degradation to technology (affects physical_constraints['ene_max_rated'] and control constraints)
            eff_e_cap = self.degraded_energy_capacity()
            # 감소된 에너지 용량을 계산
            TellUser.info(f"BATTERY - {self.name}: effective energy capacity is now {truncate_float(eff_e_cap)} kWh " +
                          f"({truncate_float(100*(1 - (self.ene_max_rated-eff_e_cap)/self.ene_max_rated), 7)}% of original)")
            # 로그를 출력하여 사용자에게 감소된 에너지 용량을 알림
            self.degrade_data.loc[opt_period, 'effective energy capacity (kWh)'] = eff_e_cap
            # 최적화 이후의 효과적인 에너지 용량을 degrade_data에 기록
            self.effective_soe_max = eff_e_cap * self.ulsoc
            self.effective_soe_min = eff_e_cap * self.llsoc
            # 효과적인 에너지0 용량에 따라 SOC의 상한과 하한을 업데이트

    def constraints(self, mask, **kwargs): # 에너지 저장 장치의 제약 조건을 생성하는 역할
        """Default build constraint list method. Used by services that do not
        have constraints.

        Args:
            mask (DataFrame): A boolean array that is true for indices
                corresponding to time_series data included in the subs data set

        Returns:
            A list of constraints that corresponds the battery's physical
                constraints and its service constraints
        """
        # create default list of constraints
        constraint_list = super().constraints(mask, **kwargs)
        # 부모 클래스의 constraints 메서드를 호출하여 기본 제약 조건 리스트를 생성
        if self.incl_binary: # 이진 변수를 포함하는지 확인하며 true인 경우에 실행
            # battery can not charge and discharge in the same timestep
            constraint_list += [cvx.NonPos(self.variables_dict['on_c'] +
                                           self.variables_dict['on_d'] - 1)]
            # 이진 변수의 제약 조건을 생성합니다. 즉, 충전 및 방전 중 하나만 활성화될 수 있도록 하는 조건입니다. 
            # CVXPY의 NonPos 함수를 사용하여 변수의 합이 1 이하인지 확인
            # 생성된 이진 변수 제약 조건을 기존의 제약 조건 리스트에 추가

        return constraint_list
        # 최종적으로 구성된 제약 조건 리스트를 반환

    def save_variable_results(self, subs_index): # 최적화 변수 결과를 저장하고, 충전 및 방전이 동시에 발생하는지 확인하는 역할
        """ Searches through the dictionary of optimization variables and saves the ones specific to each
        DER instance and saves the values it to itself

        Args:
            subs_index (Index): index of the subset of data for which the variables were solved for
        """
        super().save_variable_results(subs_index)
        # 부모 클래스의 save_variable_results 메서드를 호출하여 기본적인 최적화 변수 결과를 저장
        # check for charging and discharging in same time step
        eps = 1e-4
        # 아주 작은 값으로, 충전 및 방전이 동시에 발생하는지 확인하는 데 사용
        if np.any((self.variables_df.loc[subs_index, 'ch'].values >= eps) & (self.variables_df.loc[subs_index, 'dis'].values >= eps)):
        # 충전 및 방전 변수 중에서 0.0001 이상의 값을 가지는 경우를 찾음
        # 어떤 경우라도 참이 되는지 여부를 확인
            TellUser.warning('non-zero charge and discharge powers found in optimization solution. Try binary formulation')
            # 충전 및 방전이 동시에 발생하는 경우에 경고 메시지를 출력합니다. 이는 이진 형태로 최적화를 시도해보라는 힌트
 
    def proforma_report(self, apply_inflation_rate_func, fill_forward_func, results): 
        # 특정 가치 스트림에 대한 수익 보고서를 생성하는 메서드
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            apply_inflation_rate_func:
            fill_forward_func:
            results (pd.DataFrame):

        Returns: A DateFrame of with each year in opt_year as the index and
            the corresponding value this stream provided.

        """
        pro_forma = super().proforma_report(apply_inflation_rate_func, fill_forward_func, results)
        # 부모 클래스의 proforma_report 메서드를 호출하여 기본 proforma를 계산
        if self.hp > 0:
        # self.hp 값이 0보다 크면:
            tech_id = self.unique_tech_id()
            # 고유 기술 ID를 설정
            # the value of the energy consumed by the auxiliary load (housekeeping power) is assumed to be equal to the
            # value of energy for DA ETS, real time ETS, or retail ETS.
            optimization_years = self.variables_df.index.year.unique()
            # 최적화된 연도를 결정
            hp_proforma = pd.DataFrame()
            # housekeeping power에 대한 proforma를 저장할 빈 DataFrame을 생성
            if results.columns.isin(['Energy Price ($/kWh)']).any():
            # 결과 데이터프레임에 'Energy Price ($/kWh)' 열이 포함되어 있는지 확인
                hp_cost = self.dt * -results.loc[:, 'Energy Price ($/kWh)'] * self.hp
                # housekeeping power에 대한 비용을 계산
                for year in optimization_years:
                    # 최적화 연도에 대해 아래의 작업을 반복
                    year_monthly = hp_cost[hp_cost.index.year == year]
                    # 현재 연도에 해당하는 월별 비용을 선택
                    hp_proforma.loc[pd.Period(year=year, freq='y'), tech_id + 'Aux Load Cost'] = year_monthly.sum()
                    # proforma에 현재 연도의 housekeeping power 비용을 더함
            # fill forward
            hp_proforma = fill_forward_func(hp_proforma, None)
            # ill forward 함수를 사용하여 proforma를 채움
            # append will super class's proforma
            pro_forma = pd.concat([pro_forma, hp_proforma], axis=1)
            # housekeeping power proforma를 기본 proforma에 추가

        return pro_forma
        # 최종 proforma를 반환
        # 주어진 가치 스트림에 대한 proforma를 계산하고, housekeeping power가 존재하는 경우에는 해당 비용을 추가하여 최종 proforma를 반환

    def drill_down_reports(self, monthly_data=None, time_series_data=None, technology_summary=None, sizing_df=None):
        특정 가치 스트림에 대한 드릴다운 보고서를 생성하는 메서드
        """Calculates any service related dataframe that is reported to the user.

        Args:
            monthly_data:
            time_series_data:
            technology_summary:
            sizing_df:



        Returns: dictionary of DataFrames of any reports that are value stream specific
            keys are the file name that the df will be saved with
        """

        DCT = super().drill_down_reports(monthly_data, time_series_data, technology_summary, sizing_df)
        # 부모 클래스의 drill_down_reports 메서드를 호출하여 기본 드릴다운 보고서를 계산
        if self.incl_cycle_degrade: # 만약 incl_cycle_degrade가 true 이면
            DCT[f"{self.name.replace(' ', '_')}_degradation_data"] = self.degrade_data
            # 이전에 계산한 degradation 데이터를 딕셔너리에 추가합니다. 파일 이름은 가치 스트림의 이름에 밑줄로 대체한 것으로 설정
            total_counted_cycles = pd.concat(self.counted_cycles)
            # count된 사이클 데이터를 모두 합침
            DCT[f"{self.name.replace(' ', '_')}_cycle_counting"] = total_counted_cycles
            # 합쳐진 사이클 데이터를 딕셔너리에 추가합니다. 파일 이름은 가치 스트림의 이름에 밑줄로 대체한 것으로 설정
        return DCT
        # 최종 드릴다운 보고서 딕셔너리를 반환
        # 주어진 가치 스트림에 대한 드릴다운 보고서를 계산하고, 필요한 경우에는 degradation 데이터와 사이클 카운트 데이터를 추가하여 반환
