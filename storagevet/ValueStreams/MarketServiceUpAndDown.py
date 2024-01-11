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
MarketServiceUpAndDown.py

This Python class contains methods and attributes that help model market
services that provide service through discharging more OR charging less
relative to the power set points.
"""
from storagevet.ValueStreams.ValueStream import ValueStream
import cvxpy as cvx
import pandas as pd
import numpy as np
import storagevet.Library as Lib


class MarketServiceUpAndDown(ValueStream):
    """ 일 때 서비스를 제공할 수 있는 마켓 서비스 """
    def __init__(self, name, full_name, params):
        """ 목적 함수를 생성하고 제약 조건을 찾아 생성
        Args:
            name (str): abbreviated name
            full_name (str): the expanded name of the service
            params (Dict): input parameters
        """
        ValueStream.__init__(self, name, params)
        self.full_name = full_name
        self.combined_market = params['CombinedMarket']
        self.duration = params['duration']
        self.energy_growth = params['energyprice_growth']/100
        self.eod_avg = params['eod']
        self.eou_avg = params['eou']
        self.growth = params['growth']/100
        self.price_down = params['regd_price']
        self.price_up = params['regu_price']
        self.price_energy = params['energy_price']
        
    def get_energy_option_up(self, mask):
        """ 상향 에너지 옵션을 n x 1 벡터로 변환

        Args:
            mask:
        Returns: a CVXPY vector
        """
        return cvx.promote(self.eou_avg, mask.loc[mask].shape)

    def get_energy_option_down(self, mask):
        """ 하향 에너지 옵션을 n x 1 벡터로 변환

        Args:
            mask:
        Returns: a CVXPY vector
        """
        return cvx.promote(self.eod_avg, mask.loc[mask].shape)

    def constraints(self, mask, load_sum, tot_variable_gen, generator_out_sum,
                    net_ess_power, combined_rating):
        """최적화 엔진에 대한 제약 조건 리스트
        Args:
            mask (DataFrame): 데이터 세트에 포함된 시계열 데이터에 대응하는 인덱스에 대한 불리언 배열/ 최적화 문제에서 고려해야하는 시계열 데이터 지정
            tot_variable_gen (Expression): 가변 발전원의 총 합을 나타냄
            load_sum (list, Expression): 시스템 내의 발전의 합으로, 전체 발전을 나타냅니다.
            net_ess_power (list, Expression): 시스템 내 모든 ESS의 순 전력의 합으로, grid의 흐름이 음수인 경우 포함
            combined_rating (Dictionary):각 DER 클래스 유형의 결합 등급으로, 다양한 DER 클래스에 대한 등급 정보를 포함

        Returns:
            방정식에 추가된 최적화 변수에 대한 제약 조건 리스트
        """
        constraint_list = []
        constraint_list += [cvx.NonPos(-self.variables['up_ch'])]
        constraint_list += [cvx.NonPos(-self.variables['down_ch'])]
        constraint_list += [cvx.NonPos(-self.variables['up_dis'])]
        constraint_list += [cvx.NonPos(-self.variables['down_dis'])]
        if self.combined_market:
            constraint_list += [
                cvx.Zero(self.variables['down_dis'] + self.variables['down_ch'] -
                         self.variables['up_dis'] - self.variables['up_ch'])
            ]
# cvx.NonPos 최적화 변수들이 음수일 때 제약조건 부여 (up_ch,down_ch,up_dis,down_dis 모두 음수)/ 모든 최적화 변수의 합이 0이 되어야 하는 경우, 즉 combined_market이 True인 경우에 해당하는 제약 조건을 추가
        return constraint_list

    def p_reservation_charge_up(self, mask):
        """ "up" 방향으로 전력을 공급하는 경우에 대한 충전 전력을 이 값이 예약되어야 하는지 여부를 나타내는 CVXPY(ConVex Programming in Python) 매개변수 또는 변수를 반환합니다.
            mask (DataFrame):subs 데이터 세트에 포함된 시계열 데이터에 대응하는 인덱스에 대한 불리언 배열로, 최적화 문제에서 고려해야 하는 시계열 데이터를 지정합니다.

        Returns: CVXPY parameter/variable
        """
        return self.variables['up_ch']
# up_ch 최적화 변수 반환/ 전력을 공급하는 경우의 충전전력을 나타냄
    def p_reservation_charge_down(self, mask):
        """ "down" 방향으로 전력을 가져오는 경우에 대한 충전 전력을 이 값이 예약되어야 하는지 여부를 나타내는 CVXPY(ConVex Programming in Python) 매개변수 또는 변수를화
        Args:
            mask (DataFrame): A boolean array that is true for indices
                corresponding to time_series data included in the subs data set
        Returns:

        """
    # get_energy_option_up 및 get_energy_option_down은 다른 곳에서 정의된 메서드로,
    # 각각 "up" 및 "down" 방향의 에너지 옵션을 가져오는 데 사용됩니다.
        eou = self.get_energy_option_up(mask)
        eod = self.get_energy_option_down(mask)
    # up_ch 및 down_ch는 충전 및 방전에 사용되는 변수입니다.
    # cvx.multiply 메서드를 사용하여 각각의 변수에 에너지 옵션을 곱하고 시간 간격(dt)을 곱합니다.
        e_ch_less = cvx.multiply(self.variables['up_ch'], eou) * self.dt
        e_ch_more = cvx.multiply(self.variables['down_ch'], eod) * self.dt
    # 충전으로 인한 에너지 변화에서 방전으로 인한 에너지 변화를 뺀 값을 반환합니다.
        return e_ch_less - e_ch_more

    def uenergy_option_provided(self, mask):
        """ 방전 변경으로 인한 에너지 변화

        Args:
            mask (DataFrame): A boolean array that is true for indices
            corresponding to time_series data included in the subs data set

        Returns:
        """

        eou = self.get_energy_option_up(mask)
        eod = self.get_energy_option_down(mask)
        e_dis_less = cvx.multiply(self.variables['down_dis'], eod) * self.dt
        e_dis_more = cvx.multiply(self.variables['up_dis'], eou) * self.dt
     # 방전으로 인한 에너지 변화에서 충전으로 인한 에너지 변화를 뺀 값을 반환합니다.
        return e_dis_more - e_dis_less

    def worst_case_uenergy_stored(self, mask):
        """ 현재 SOE에서 시작하여, 시간 단계 간의 시계열에서 포착되지 않는 시간 간격 사이에서
    발생할 수 있는 위반을 방지하기 위해 이 값 스트림에 대한 예약해야 하는 에너지의 양을 계산합니다.

        Note: stored energy should be positive and provided energy should be
            negative

        Args:
            mask (DataFrame): A boolean array that is true for indices
                corresponding to time_series data included in the subs data set

        Returns: tuple (stored, provided),
            where the first value is the case where the systems would end up
            with more energy than expected and the second corresponds to the
            case where the systems would end up with less energy than expected

        """
        stored \
            = self.variables['down_ch'] * self.duration \
            + self.variables['down_dis'] * self.duration
    # down_ch: charging power capacity for frequency regulation
    # down_dis: discharging power capacity for frequency regulation
        return stored

    def worst_case_uenergy_provided(self, mask):
        """ 현재 SOE에서 이 값 스트림에 예약해야 하는 에너지 양/ 저장된 에너지는 양수이어야하며 제공된 에너지는 음수.

        Args:
            mask (DataFrame): 서브 데이터 세트에 포함된 시계열 데이터에 해당하는 인덱스에 대한 불리언 배열.

        Returns: tuple (stored, provided),
          첫 번째 값은 시스템이 예상보다 더 많은 에너지를 가지게 될 경우이며,
            두 번째 값은 시스템이 예상보다 더 적은 에너지를 가지게 될 경우임
        """
        provided \
            = self.variables['up_ch'] * -self.duration \
            + self.variables['up_dis'] * -self.duration
        return provided

    def timeseries_report(self):
        """ 이 Value Stream에 대한 최적화 결과를 요약

        Returns: A timeseries dataframe with user-friendly column headers that
            summarize the results pertaining to this instance

        """
 # 결과를 저장할 빈 데이터프레임 생성
        report = pd.DataFrame(index=self.price_energy.index)
        # GIVEN
        report.loc[:, f"{self.name} Up Price ($/kW)"] \
            = self.price_up
        report.loc[:, f"{self.name} Down Price ($/kW)"] \
            = self.price_down
        report.loc[:, f"{self.name} Energy Settlement Price ($/kWh)"] = \
            self.price_energy

        # OPTIMIZATION VARIABLES
        report.loc[:, f'{self.full_name} Down (Charging) (kW)'] \
            = self.variables_df['down_ch']
        report.loc[:, f'{self.full_name} Down (Discharging) (kW)'] \
            = self.variables_df['down_dis']
        report.loc[:, f'{self.full_name} Up (Charging) (kW)'] \
            = self.variables_df['up_ch']
        report.loc[:, f'{self.full_name} Up (Discharging) (kW)'] \
            = self.variables_df['up_dis']

        # CALCULATED EXPRESSIONS (ENERGY THROUGH-PUTS)
        e_thru_down_dis = np.multiply(self.eod_avg,
                                      self.variables_df['down_dis']) * self.dt
        e_thru_down_ch = np.multiply(self.eod_avg,
                                     self.variables_df['down_ch']) * self.dt
        e_thru_up_dis = -np.multiply(self.eou_avg,
                                     self.variables_df['up_dis']) * self.dt
        e_thru_up_ch = -np.multiply(self.eou_avg,
                                    self.variables_df['up_ch']) * self.dt
        uenergy_down = e_thru_down_dis + e_thru_down_ch
        uenergy_up = e_thru_up_dis + e_thru_up_ch

        column_start = f"{self.name} Energy Throughput"
     # 에너지 스루풋 및 상세 정보 추가
        report.loc[:, f"{column_start} (kWh)"] = uenergy_down + uenergy_up
        report.loc[:, f"{column_start} Up (Charging) (kWh)"] = e_thru_up_ch
        report.loc[:, f"{column_start} Up (Discharging) (kWh)"] = e_thru_up_dis
        report.loc[:, f"{column_start} Down (Charging) (kWh)"] = e_thru_down_ch
        report.loc[:, f"{column_start} Down (Discharging) (kWh)"] \
            = e_thru_down_dis

        return report

    def proforma_report(self, opt_years, apply_inflation_rate_func, fill_forward_func, results):
        """ 이 value stream에 참여하는 proforma 계산
        Args:
            opt_years (list): 최적화 문제가 실행된 연도 목록
            apply_inflation_rate_func: 인플레이션 비율을 적용하는 함수
            fill_forward_func: 앞으로 채워넣는 함
            results (pd.DataFrame): 최적화 변수 솔루션을 포함하는 데이터프레

        Returns: 연도별로 인덱싱된 DataFrame (이 스트림이 제공한 해당 값이 포함됨)
        """
      # 부모 클래스의 proforma_report 메서드 호출
        proforma = super().proforma_report(opt_years, apply_inflation_rate_func,
                                           fill_forward_func, results)
      # 각각의 가치 스트림에 대한 수익 계산
        pref = self.full_name
        reg_up = \
            results.loc[:, f'{pref} Up (Charging) (kW)'] \
            + results.loc[:, f'{pref} Up (Discharging) (kW)']
        regulation_up_prof = np.multiply(reg_up, self.price_up)

        reg_down = \
            results.loc[:, f'{pref} Down (Charging) (kW)'] \
            + results.loc[:, f'{pref} Down (Discharging) (kW)']
        regulation_down_prof = np.multiply(reg_down, self.price_down)
     # 에너지 스루풋 계산

        # NOTE: TODO: here we use rte_list[0] wqhich grabs the first available rte from an active ess
        #   we will want to change this to actually use all available rte values from the list
        energy_throughput = \
            results.loc[:, f"{self.name} Energy Throughput Down (Charging) (kWh)"] / self.rte_list[0] \
            + results.loc[:, f"{self.name} Energy Throughput Down (Discharging) (kWh)"] \
            + results.loc[:, f"{self.name} Energy Throughput Up (Charging) (kWh)"] / self.rte_list[0] \
            + results.loc[:, f"{self.name} Energy Throughput Up (Discharging) (kWh)"]
        energy_through_prof = np.multiply(energy_throughput, self.price_energy)

        # 모든 value stream을 하나의 데이터프레임으로 결합
        #   splicing into years
        fr_results = pd.DataFrame({'E': energy_through_prof,
                                   'RU': regulation_up_prof,
                                   'RD': regulation_down_prof},
                                  index=results.index)
        market_results_only = proforma.copy(deep=True)
     # 연도별로 계산된 수익을 프로포마에 추가
        for year in opt_years:
            year_subset = fr_results[fr_results.index.year == year]
            yr_pd = pd.Period(year=year, freq='y')
            proforma.loc[yr_pd, f'{self.name} Energy Throughput'] \
                = -year_subset['E'].sum()
            market_results_only.loc[yr_pd, f'{pref} Up'] \
                = year_subset['RU'].sum()
            market_results_only.loc[yr_pd, f'{pref} Down'] \
                = year_subset['RD'].sum()
        # 성장률에 해당하는 인플레이션을 사용하여 성장 열을 앞으로 채우기
        market_results_only = fill_forward_func(market_results_only, self.growth)
        proforma = fill_forward_func(proforma, self.energy_growth)
       # 두 데이터프레임을 합치기
        proforma = pd.concat([proforma, market_results_only], axis=1)
        return proforma
