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
MarketServiceUp.py

This Python class contains methods and attributes specific for service analysis within StorageVet.
"""
from storagevet.ValueStreams.ValueStream import ValueStream
import numpy as np
import cvxpy as cvx
import pandas as pd
import storagevet.Library as Lib

# 에너지 저장 시스템이 참여하는 시장 서비스를 나타내며, 최적화 문제의 목적 함수 및 제약 조건을 생성하고 관리하는 데 사용됩니다.

class MarketServiceUp(ValueStream):
    """MarketServiceUp 클래스는 ValueStream 클래스를 상속받아 에너지 저장 시스템이 참여하는 특정 시장 서비스를 표현합니다.

    """

    def __init__(self, name, full_name, params):
        """객체 생성 시 호출되는 초기화 메서드로, 클래스의 속성들을 초기화합니다.
        name, full_name, params를 매개변수로 받아와서 상위 클래스 ValueStream의 생성자를 호출하여 기본 속성을 초기화합니다.
        params 딕셔너리에서 price, growth, duration 등의 매개변수를 추출하여 해당 클래스 속성으로 설정합니다.
        self.variable_names은 {'ch_less', 'dis_more'}로 설정되어 있습니다.
        빈 DataFrame인 self.variables_df를 생성하고, 열은 self.variable_names에 따라 초기화됩니다.
        Args:
            name (str): abbreviated name
            full_name (str): the expanded name of the service
            params (Dict): input parameters
        """
        ValueStream.__init__(self, name, params)
        self.price = params['price']       
        self.growth = params['growth']/100  # growth rate of spinning reserve price (%/yr)
        self.duration = params['duration']  
        self.full_name = full_name
        self.variable_names = {'ch_less', 'dis_more'}
        self.variables_df = pd.DataFrame(columns=self.variable_names)

    def grow_drop_data(self, years, frequency, load_growth):
     # 주어진 데이터를 성장시키거나 추가된 데이터를 제거 /주로 시뮬레이션 분석에 사용되는 데이터를 처리하는 목적으로 설계
        """ Lib.fill_extra_data 함수를 사용하여 self.price 속성에 대한 주어진 연도 및 성장률을 사용하여 추가 데이터를 성장시킵니다.
        Lib.drop_extra_data 함수를 사용하여 self.price 속성에서 주어진 연도에 해당하는 데이터를 제거합니다.
        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this
                simulation

        """
        self.price = Lib.fill_extra_data(self.price, years, self.growth, frequency)  # 시장 서비스의 price 속성에 대해 주어진 연도와 성장률을 사용하여 추가 데이터를 성장
        self.price = Lib.drop_extra_data(self.price, years)                          # 시장 서비스의 price 속성에서 주어진 연도에 해당하는 데이터를 제거

    def initialize_variables(self, size):
     # 최적화에 필요한 변수들을 초기화하고 딕셔너리에 추가 /주로 CVXPY 라이브러리를 사용하여 최적화에 필요한 변수를 정의하고 이를 클래스의 속성으로 저장
        """cvx.Variable 함수를 사용하여 ch_less와 dis_more 두 가지 최적화 변수를 생성합니다.
        이 변수들은 CVXPY 라이브러리에서 제공하는 최적화 변수를 나타냅니다.
        생성된 변수들은 self.variables 딕셔너리에 저장되어 클래스의 속성으로 활용됩니다.

        Variables added:
            dis_more (Variable): A cvxpy variable for spinning reserve capacity to increase
                discharging power
            ch_less (Variable): A cvxpy variable for spinning reserve capacity to decrease
                charging power

        Args:
            size (Int): Length of optimization variables to create

        Returns:
            Dictionary of optimization variables
        """
        self.variables = {'ch_less': cvx.Variable(shape=size, name=f'{self.name}_ch_less'),    
                          'dis_more': cvx.Variable(shape=size, name=f'{self.name}_dis_more')}
     # CVXPY라이브러리를 사용하여 최적화 변수(ch_less,dis_more)를 self.variables에 저장

    def objective_function(self, mask, load_sum, tot_variable_gen, generator_out_sum,
                           net_ess_power, annuity_scalar=1):
        """ 적화의 목적 함수를 생성하는 역할을 합니다. 
            함수는 주어진 데이터와 변수들을 기반으로 CVXPY 라이브러리를 사용하여 최적화 목적 함수를 정의하고 반환
            cvx.Parameter 함수를 사용하여 payment 파라미터를 생성합니다. 
            이는 시간에 따른 가격을 나타내며, self.price에서 mask에 해당하는 값을 사용합니다.
            
            최종적으로, 최적화 목적 함수를 정의하고 반환합니다. 
            목적 함수는 시간에 따른 payment와 self.variables['ch_less'], self.variables['dis_more']를 사용하여 정의됩니다.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series
                data included in the subs data set
            tot_variable_gen (Expression): the sum of the variable/intermittent generation sources
            load_sum (list, Expression): the sum of load within the system
            generator_out_sum (list, Expression): the sum of conventional generation within the
                system
            net_ess_power (list, Expression): the sum of the net power of all the ESS in the
                system. [= charge - discharge]
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit
                that helps capture the cost/benefit over the entire project lifetime (only to be
                set iff sizing)

        Returns:
            The expression of the objective function that it affects. This can be passed into the
            cvxpy solver.

        """
        payment = cvx.Parameter(sum(mask), value=self.price.loc[mask].values,
                                name=f'{self.name}_price')
# CVXPY 라이브러리를 사용하여 payment를 생성/ payment는 시간에 따른 가격을 나타냄
        return {
            self.name: cvx.sum(
                cvx.multiply(-payment, self.variables['ch_less']) +
                cvx.multiply(-payment, self.variables['dis_more'])) * self.dt * annuity_scalar}
# CVXPY 라이브러리를 사용하여 최적화 변수와 가격을 사용하여 최종 목적 함수를 정의
    def constraints(self, mask, load_sum, tot_variable_gen, generator_out_sum, net_ess_power,
                    combined_rating):
        """ 제약 조건의 목록을 생성하는 역할을 합니다. 
            주어진 데이터와 변수들을 기반으로 CVXPY 라이브러리를 사용하여 최적화에 적용될 제약 조건들을 정의하고 목록으로 반환

            빈 제약 조건 목록을 생성합니다.
            cvx.NonPos(-self.variables['ch_less']): self.variables['ch_less'] 변수가 음수가 되지 않도록 하는 제약 조건을 생성합니다. 
            이는 충전 용량을 음수로 제한하려는 것입니다.
            
            cvx.NonPos(-self.variables['dis_more']): self.variables['dis_more'] 변수가 음수가 되지 않도록 하는 제약 조건을 생성합니다. 
            이는 방전 용량을 음수로 제한하려는 것입니다.
            
            생성된 제약 조건들을 목록에 추가하고 반환합니다.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series
                data included in the subs data set
            tot_variable_gen (Expression): the sum of the variable/intermittent generation sources
            load_sum (list, Expression): the sum of load within the system
            generator_out_sum (list, Expression): the sum of conventional generation within
                the system
            net_ess_power (list, Expression): the sum of the net power of all the ESS in the
                system. flow out into the grid is negative
            combined_rating (Dictionary): the combined rating of each DER class type

        Returns:
            constraint_list (list): list of constraints

        """
        constraint_list = []
        constraint_list += [cvx.NonPos(-self.variables['ch_less'])]        # self.variables['ch_less'] 변수가 음수가 되도록 하는 제약 조건을 생성/이는 충전 용량을 음수로 제한하려는 의도
        constraint_list += [cvx.NonPos(-self.variables['dis_more'])]       # self.variables['dis_more'] 변수가 음수가 되도록 하는 제약 조건을 생성/이는 방 용량을 음수로 제한하려는 의도
        return constraint_list

    def p_reservation_charge_up(self, mask):
        """ 해당 시장 서비스에서 사용되는 충전 전력의 양을 반환하는 역할을 합니다. 
            반환되는 값은 CVXPY 라이브러리에서 사용되는 파라미터 또는 변수입니다.
            

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series
                data included in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.variables['ch_less']

    def p_reservation_discharge_up(self, mask):
        """ 해당 시장 서비스에서 사용되는 방전 전력의 양을 반환하는 역할을 합니다. 
            반환되는 값은 CVXPY 라이브러리에서 사용되는 파라미터 또는 변수입니다.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series
                data included in the subs data set

        Returns: CVXPY parameter/variable

        """
        return self.variables['dis_more']

    def worst_case_uenergy_provided(self, mask):
        """ 재 SOE(에너지 저장 장치의 상태)에서 해당 시장 서비스에 대한 에너지 예약을 계산합니다. 
            반환값은 두 가지 경우에 대한 튜플이며, 각각은 에너지가 예상보다 많아지는 경우와 적어지는 경우를 나타냅니다.

            provided: 현재 시간 텀 내에서 self.variables['ch_less']와 self.variables['dis_more']에 의해 제공되는 에너지를 계산합니다. 
            ch_less는 양수일 때 많은 에너지 공급을 나타내며, dis_more는 양수일 때 적은 에너지 공급을 나타냅니다.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series
                data included in the subs data set

        Returns: tuple (stored, provided),
            where the first value is the case where the systems would end up with more energy than
            expected and the second corresponds to the case where the systems would end up with
            less energy than expected

        """
        provided = self.variables['ch_less']*self.duration + self.variables['dis_more']*self.duration # 현재 시간 텀 내에서 충전 및 방전에 의해 제공되는 에너지를 나타냄/ 음수일 때 많은 에너지 공급. 양수일 때 적은 에너지 공
        return provided

    def timeseries_report(self):
        """ 최적화 결과를 요약하고 사용자에게 보기 쉬운 형식으로 제공하는 역할을 합니다. 
            반환값은 시계열 데이터프레임이며, 해당 Value Stream 인스턴스와 관련된 결과를 나타내는 사용자 친화적인 열 헤더를 포함합니다.

            report 데이터프레임을 생성하고, 인덱스는 self.price.index로 설정됩니다.
            
            다양한 열이 추가됩니다.
            "{self.name} Price ($/kW)": ValueStream의 가격 정보를 나타냅니다.
            "{self.full_name} Up (Charging) (kW)": ValueStream의 충전 전력량을 나타냅니다.
            "{self.full_name} Up (Discharging) (kW)": ValueStream의 방전 전력량을 나타냅니다.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the
            results pertaining to this instance

        """
        report = pd.DataFrame(index=self.price.index)
        report.loc[:, f"{self.name} Price ($/kW)"] = self.price        # self.price를 report 데이터 프레임에 추가 ValueStream의 가격정보를 나타냄
        report.loc[:, f"{self.full_name} Up (Charging) (kW)"] = self.variables_df['ch_less'] # self.price를 report 데이터 프레임에 추가 ValueStream의 충전 전력량을 나타냄
        report.loc[:, f"{self.full_name} Up (Discharging) (kW)"] = self.variables_df['dis_more'] # self.price를 report 데이터 프레임에 추가 ValueStream의 방전 전력량를 나타냄

        return report

    def proforma_report(self, opt_years, apply_inflation_rate_func, fill_forward_func, results):
        """ 특정 시장 서비스에 참여한 결과에 기반하여 해당 가치 스트림에 대한 수익 예측을 계산합니다. 
            이러한 계산은 연간으로 진행되며, 수익은 최적화 결과와 시장 가격을 고려하여 산출됩니다.

            super().proforma_report를 호출하여 상위 클래스의 proforma_report 메서드를 실행하고 초기화합니다.
            
            results 데이터프레임으로부터 "{self.full_name} Up (Charging) (kW)"와 "{self.full_name} Up (Discharging) (kW)" 열을 사용하여 입찰(bid)을 계산합니다. 
            이는 충전 및 방전 전력량을 합산한 결과입니다.
            
            bid와 가격 정보를 이용하여 일정 시간 동안의 비용 또는 수익을 계산합니다.
            
            각 연도별로 계산된 수익을 proforma 데이터프레임에 추가합니다.
            
            fill_forward_func를 사용하여 성장률이 적용된 컬럼을 전방으로 채웁니다.

        Args:
            opt_years (list): list of years the optimization problem ran for
            apply_inflation_rate_func:
            fill_forward_func:
            results (pd.DataFrame): DataFrame with all the optimization variable solutions

        Returns: A DateFrame (of with each year in opt_year as the index and the corresponding
        value this stream provided)
        """
        proforma = super().proforma_report(opt_years, apply_inflation_rate_func,
                                           fill_forward_func, results)
        bid = \
            results.loc[:, f'{self.full_name} Up (Charging) (kW)'] + \
            results.loc[:, f'{self.full_name} Up (Discharging) (kW)']
        spinning_prof = np.multiply(bid, self.price) * self.dt
# bid는 충전 및 방전 전력량을 합산한 결과를 나타내는 변수/ bid와 가격 정보를 기반으로 일정 시간동안의 비용 또는 수익 계산(전력요금)
        for year in opt_years:
            year_subset = spinning_prof[spinning_prof.index.year == year]
            proforma.loc[pd.Period(year=year, freq='y'), self.full_name] = year_subset.sum()
        # forward fill growth columns with inflation at growth rate
        proforma = fill_forward_func(proforma, self.growth)
# 각 연도에 대해 proforma를 계산하고, 해당 연도에 대한 결과를 proforma 데이터 프레임에 추
        return proforma

    def update_price_signals(self, monthly_data, time_series_data):
        """ Updates attributes related to price signals with new price signals that are saved in
        the arguments of the method. Only updates the price signals that exist, and does not
        require all price signals needed for this service.

        Args:
            monthly_data (DataFrame): monthly data after pre-processing
            time_series_data (DataFrame): time series data after pre-processing

        """
        try:
            self.price = time_series_data.loc[:, f'{self.name} Price ($/kW)']
        except KeyError:
            pass
# 시계열 데이터에서 새로운 가격 신호를 업데이트하고 만약 해당 데이터가 없다면 예외를 처리하고 무시
