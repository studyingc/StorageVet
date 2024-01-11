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
LoadFollowing.py

This Python class contains methods and attributes specific for service analysis
within StorageVet.
"""
from storagevet.ValueStreams.MarketServiceUpAndDown import MarketServiceUpAndDown
import cvxpy as cvx
import numpy as np
import storagevet.Library as Lib
from storagevet.ErrorHandling import *


class LoadFollowing(MarketServiceUpAndDown):
    """ Load Following.

    """

    def __init__(self, params):
        """ Generates the objective function, finds and creates constraints.

        Args:
            params (Dict): input parameters
        """
        MarketServiceUpAndDown.__init__(self, 'LF', 'Load Following', params)    """MarketServiceUpAndDown 클래스의 생성자를 호출하여 해당 클래스의 초기화를 먼저 수행/params라는 딕셔너리 형태의 입력 매개변수를 받음"""
        self.u_ts_constraints = params.get('u_ts_constraints', False)            """u_ts_constraints 속성 설정"""
        self.d_ts_constraints = params.get('d_ts_constraints', False)            """d_ts_constraints 속성 설정"""
        if self.u_ts_constraints:
            self.regu_max = params['lf_u_max']
            self.regu_min = params['lf_u_min']
        if self.d_ts_constraints:
            self.regd_max = params['lf_d_max']
            self.regd_min = params['lf_d_min']

        if self.dt > 0.25:                                                    """dt가 0.25보다 클 경우 경고메시지 출력"""
            TellUser.warning("WARNING: using Load Following Service and " +
                             "time series timestep is greater than 15 min.")

    def grow_drop_data(self, years, frequency, load_growth):
        """ Adds data by growing the given data OR drops any extra data that
        might have slipped in. Update variable that hold timeseries data
        after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of
                loads in this simulation

        """
        super().grow_drop_data(years, frequency, load_growth)                     """부모 클래스인 MarketServiceUpAndDown의 grow_drop_data 메서드를 호출하여 해당 메서드 먼저 실행"""
        self.eou_avg = Lib.fill_extra_data(self.eou_avg, years, 0, frequency)     """eou_avg변수에 대해 Lib.fill_extra_data 함수를 사용하여 추가된 데이터를 0으로 채움"""
        self.eou_avg = Lib.drop_extra_data(self.eou_avg, years)                   """eou_avg 변수에 대해 Lib.drop_extra_data 함수를 사용하여 추가된 데이터 중 불필요한 데이터를 삭제"""

        self.eod_avg = Lib.fill_extra_data(self.eod_avg, years, 0, frequency)
        self.eod_avg = Lib.drop_extra_data(self.eod_avg, years)

        if self.u_ts_constraints:
            self.regu_max = Lib.fill_extra_data(self.regu_max, years,
                                                0, frequency)
            self.regu_max = Lib.drop_extra_data(self.regu_max, years)

            self.regu_min = Lib.fill_extra_data(self.regu_min, years,
                                                0, frequency)
            self.regu_min = Lib.drop_extra_data(self.regu_min, years)

        if self.d_ts_constraints:
            self.regd_max = Lib.fill_extra_data(self.regd_max, years,
                                                0, frequency)
            self.regd_max = Lib.drop_extra_data(self.regd_max, years)

            self.regd_min = Lib.fill_extra_data(self.regd_min, years,
                                                0, frequency)
            self.regd_min = Lib.drop_extra_data(self.regd_min, years)             """그 과정 반복"""

    def get_energy_option_up(self, mask):                      
        """ transform the energy option up into a n x 1 vector

        Args:
            mask:

        Returns: a CVXPY vector

        """
        return cvx.Parameter(sum(mask), value=self.eou_avg.loc[mask].values,
                             name='LF_EOU')                                     """'mask'에 해당하는 시점들의 'eou_avg'데이터를 사용하여 CVXPY의 파라미터 생성 (상향 에너지 옵션)"""

    def get_energy_option_down(self, mask):
        """ transform the energy option down into a n x 1 vector

        Args:
            mask:

        Returns: a CVXPY vector

        """
        return cvx.Parameter(sum(mask), value=self.eod_avg.loc[mask].values,
                             name='LF_EOD')                                   """'mask'에 해당하는 시점들의 'eod_avg'데이터를 사용하여 CVXPY의 파라미터 생성 (하향 에너지 옵션)"""

    def constraints(self, mask, load_sum, tot_variable_gen, generator_out_sum,
                    net_ess_power, combined_rating):
        """build constraint list method for the optimization engine

        Args:
            mask (DataFrame): A boolean array that is true for indices
                corresponding to time_series data included in the subs data set
            tot_variable_gen (Expression): the sum of the variable/intermittent
                generation sources
            load_sum (list, Expression): the sum of load within the system
            generator_out_sum (list, Expression): the sum of conventional
                generation within the system
            net_ess_power (list, Expression): the sum of the net power of all
                the ESS in the system. flow out into the grid is negative
            combined_rating (Dictionary): the combined rating of each DER class
                type

        Returns:
            An list of constraints for the optimization variables added to
            the system of equations
        """
        constraint_list = super().constraints(mask, load_sum, tot_variable_gen,
                                              generator_out_sum,
                                              net_ess_power, combined_rating)        """상위 클래스인 MarketServiceUpAndDown의 constraints 메서드를 호출하여 초기 제약 조건 목록을 가져옴"""

        # add time series service participation constraints, if called for
        #   Reg Up Max and Reg Up Min will constrain the sum of up_ch + up_dis
        if self.u_ts_constraints:
            constraint_list += [
                cvx.NonPos(self.variables['up_ch'] + self.variables['up_dis']
                           - self.regu_max.loc[mask])
            ]                                                                       """상향 참여의 제약 조건으로, up_ch와 up_dis의 합이 regu_max를 초과하지 않아야 함"""
            constraint_list += [
                cvx.NonPos((-1) * self.variables['up_ch'] + (-1) * self.variables[
                    'up_dis'] + self.regu_min.loc[mask])
            ]                                                                       """상향 참여의 제약 조건으로, up_ch와 up_dis의 합이 regu_min 미만이어야 함"""
        #   Reg Down Max and Reg Down Min will constrain the sum down_ch+down_dis
        if self.d_ts_constraints:
            constraint_list += [
                cvx.NonPos(self.variables['down_ch'] + self.variables['down_dis']
                           - self.regd_max.loc[mask])
            ]                                                                       """하향 참여의 제약 조건으로, down_ch와 down_dis의 합이 regd_max를 초과하지 않아야 함"""
            constraint_list += [
                cvx.NonPos(-self.variables['down_ch'] - self.variables['down_dis']
                           + self.regd_min.loc[mask])
            ]                                                                        """하향 참여의 제약 조건으로, down_ch와 down_dis의 합이 regd_min를 미만이어야 함"""
        return constraint_list  """최적화 엔진에 추가할 모든 제약 조건을 구축하고 반환"""

    def update_price_signals(self, monthly_data, time_series_data):
        """ Updates attributes related to price signals with new price signals
        that are saved in the arguments of the method. Only updates the
        price signals that exist, and does not require all price signals
        needed for this service.

        Args:
            monthly_data (DataFrame): monthly data after pre-processing
            time_series_data (DataFrame): time series data after pre-processing

        """
        if self.combined_market:                                      
            try:
                fr_price = time_series_data.loc[:, 'LF Price ($/kW)']         """LF Price ($/kW) 열을 시계열 데이터로부터 가져와 fr_price변수에 할당"""
            except KeyError:
                pass
            else: 
                self.p_regu = np.divide(fr_price, 2)                         """LF Price($/kW)값을 2로나눈 값을 self.p_regu에 할당"""
                self.p_regd = np.divide(fr_price, 2)                         """LF Price($/kW)값을 2로나눈 값을 self.p_regd에 할당"""

            try:
                self.price = time_series_data.loc[:, 'DA Price ($/kWh)']     """DA Price ($/kWh)열을 시계열 데이터로부터 가져와 self.price에 할"""
            except KeyError:
                pass
        else:
            try:
                self.p_regd = time_series_data.loc[:, 'LF Down Price ($/kW)']  """LF Down Price ($/kW)열을 시계열 데이터로부터 가져와 self.p_regd에 할당"""
            except KeyError:                                            
                pass

            try:
                self.p_regu = time_series_data.loc[:, 'LF Up Price ($/kW)']    """LF Up Price ($/kW) 열을 시계열 데이터로부터 가져와 self.p_regu에 할당"""
            except KeyError:
                pass

            try:
                self.price = time_series_data.loc[:, 'DA Price ($/kWh)']      """DA Price ($/kWh)' 열을 시계열 데이터로부터 가져와 self.price에 할당"""
            except KeyError:
                pass

    def min_regulation_up(self):
        if self.u_ts_constraints:                 # 상향 시계열 제약이 활성화 된 경우
            return self.regu_min                  # regu_min 속성을 반환
        return super().min_regulation_up()        # 상향 시계열 제약이 비활성화 된 경우 부모 클래스인 MarketServiceUpAndDown의 min_regulation_up 메서드를 호출하여 해당 값 반환

    def min_regulation_down(self):               
        if self.d_ts_constraints:                 # 하향 시계열 제약이 활성화 된 경우
            return self.regd_min                  # regd_min 속성을 반환
        return super().min_regulation_down()      # 하향 시계열 제약이 비활성화 된 경우 부모 클래스인 MarketServiceUpAndDown의 min_regulation_down 메서드를 호출하여 해당 값을 반환

    def max_participation_is_defined(self):
        return self.u_ts_constraints and self.d_ts_constraints  # 두 시계열 제약이 모두 정의 되었으면 True, 그렇지 않으면 False를 반환
