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
Technology

This Python class contains methods and attributes specific for technology analysis within StorageVet.
"""
import pandas as pd
import numpy as np
import cvxpy as cvx
from storagevet.ErrorHandling import *


class DER:
    """ A general template for DER object, which could be any kind of Distributed Energy Resources currently
        supported in DERVET: storage (CAES, Battery), generator (CHP, ICE), renewable (PV), and loads


    """

    def __init__(self, params):
        """ Initialize all technology with the following attributes.

        Args:
            params (dict): Dict of parameters
        """
        TellUser.debug(f"Initializing {__name__}")
       
        # initialize internal attributes
        self.name = params['name']  # specific tech model name
        # 기술 모델의 이름을 초기화
        self.dt = params['dt']
        # 시간 간격을 초기화
        #self.technology_type = None  # "Energy Storage System", "Rotating Generator", "Intermittent Resource", "Load"
        #self.tag = None
        self.variable_om = 0  # $/kWh
        # 변수 O&M 비용을 0으로 초기화
        self.id = params.get('ID')
        # ID를 초기화

        # attributes about specific to each DER
        self.variables_df = pd.DataFrame()  # optimization variables are saved here
        # 최적화 변수가 여기에 저장될 데이터 프레임을 초기화
        self.variables_dict = {}  # holds the CVXPY variables upon creation in the technology instance
        # CVXPY 변수가 기술 인스턴스에서 생성될 때 이곳에 저장

        # boolean attributes
        self.is_electric = False    # can this DER consume or generate electric power?
        self.is_hot = False         # can this DER consume or generate heat?
        self.is_cold = False        # can this DER consume or generate cooling power?
        self.is_fuel = False        # can this DER consume fuel?
        # 각각 DER가 전기, 열, 냉각, 연료를 생성하거나 소비할 수 있는지 여부를 나타내는 불리언 속성을 초기화

        self.can_participate_in_market_services = True
        # 시장 서비스에 참여할 수 있는지 여부를 나타내는 불리언 속성을 초기화

    def set_fuel_cost(self, function_pointer):
        # DER가 연료를 소비할 수 있는 경우 fuel_cost 속성을 설정하는 데 사용
        """
        Sets a fuel_cost attribute if the DER can consume fuel

        Args:
            function_pointer (function): likely to be Financial.get_fuel_costs()
        """
        if self.is_fuel:
        # ER가 연료를 소비할 수 있는 경우에만 아래 코드 블록을 실행
            self.fuel_cost = function_pointer(self.fuel_type)
            # function_pointer를 사용하여 연료 비용을 설정
            TellUser.debug(f'Setting a fuel_cost attribute: {self.fuel_cost} $/MMBtu (fuel_type: {self.fuel_type}) for this DER: {self.tag}: {self.name}')
            # 디버그 정보를 출력하여 연료 비용이 어떻게 설정되었는지에 대한 정보를 제공
    def zero_column_name(self):
    # 프로포르마 생성에 사용되는 특정 기술의 'Capital Cost' 열 이름을 반환
        return self.unique_tech_id() + ' Capital Cost'  # used for proforma creation
        # 해당 기술의 고유 ID에 ' Capital Cost'를 추가하여 'Capital Cost' 열의 이름을 생성 및 반환

    def fixed_column_name(self):
    # 프로포르마 생성에 사용되는 특정 기술의 'Fixed O&M Cost' 열 이름을 반환
        return self.unique_tech_id() + ' Fixed O&M Cost'  # used for proforma creation
        # 해당 기술의 고유 ID에 ' Fixed O&M Cost'를 추가하여 'Fixed O&M Cost' 열의 이름을 생성 및 반환

    def get_capex(self, **kwargs) -> cvx.Variable or float:
    # 해당 DER의 CAPEX(자본비용)를 반환
        """

        Returns: the capex of this DER

        """
        return 0
        # DER의 CAPEX를 구하는 방식이 특별한 로직이나 계산을 필요로하지 않는 경우에 사용가능

    def grow_drop_data(self, years, frequency, load_growth):
        # 주어진 데이터를 성장시키거나 추가로 포함된 데이터를 삭제하여 데이터를 업데이트합니다. 
        # 이 메서드는 add_growth_data 후에 호출되어야 하며, 최적화가 실행되기 전에 호출되어야 함
        """ Adds data by growing the given data OR drops any extra data that might have slipped in.
        Update variable that hold timeseries data after adding growth data. These method should be called after
        add_growth_data and before the optimization is run.

        Args:
            years (List): list of years for which analysis will occur on
            frequency (str): period frequency of the timeseries data
            load_growth (float): percent/ decimal value of the growth rate of loads in this simulation

        """
        pass

    def discharge_capacity(self):
    # 최대 배출 용량을 반환합니다. 현재의 구현에서는 항상 0을 반환
        """

        Returns: the maximum discharge that can be attained

        """
        return 0

    def charge_capacity(self):
    #  최대 충전용량을 반환
        """

        Returns: the maximum charge that can be attained

        """
        return 0

    def operational_max_energy(self):
    # 사용자 입력에 기반하여 이 DER에 저장될 수 있는 최대 에너지를 반환
        """

        Returns: the maximum energy that should stored in this DER based on user inputs

        """

        return 0

    def operational_min_energy(self):
    # 사용자 입력에 기반하여 이 DER에 저장되어야 하는 최소 에너지를 반환
    
        """

        Returns: the minimum energy that should stored in this DER based on user inputs
        """

        return 0

    def qualifying_capacity(self, event_length):
    # RA(신뢰성 증진) 또는 DR(수요 응답) 이벤트에 참여하기 위해 DER이 발전할 수 있는 전력을 설명
    # 시스템의 자격 취득을 결정하는 데 사용
        """ Describes how much power the DER can discharge to qualify for RA or DR. Used to determine
        the system's qualifying commitment.

        Args:
            event_length (int): the length of the RA or DR event, this is the
                total hours that a DER is expected to discharge for

        Returns: int/float

        """
        return 0

    def initialize_variables(self, size):
        """ Adds optimization variables to dictionary

        Variables added:

        Args:
            size (Int): Length of optimization variables to create

        """
        pass

    def get_state_of_energy(self, mask):
    # 에너지 상태를 나타내는 파라미터를 생성하는 역할
    # 시계열 데이터가 있는 인덱스에 대해 ture 인 부울 배열
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the state of energy as a function of time for the

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}-Zero')
        # value는 크기가 sum(mask)와 동일하고 초기값 0으로 채워짐(ture의 개수와 동일
        # mask 에 따라서 에너지 상태를 나타내는 파라미터를 생성하고 이를 반환

    def get_discharge(self, mask):
    # 시간에 따른 방전을 나타냄
    # 주어진 mask를 기반으로 시간에 따른 방전을 나타내는 cvx.Parameter 객체를 반환
        """ The effective discharge of this DER
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the discharge as a function of time for the

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}-Zero')
        # 볼록 최적화 문제에서 사용하기 위한 파라미터 객체를 생성
        # 반환된 파라미터는 mask를 기반으로 시간에 따른 방전을 나타냄
 
    def get_charge(self, mask):
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the charge as a function of time for the

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}-Zero')
        # 반환된 파라미터는 mask를 기반으로 시간에 따른 충전을 나타냄

    def get_net_power(self, mask):
    # 시간에 따른 순전ㄹ녁을 나타내며, get_charge와 get_discharge 메서드를 사용하여 충,방전을 계산하여 반환
        """
        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                in the subs data set

        Returns: the net power [= charge - discharge] as a function of time for the

        """
        return self.get_charge(mask) - self.get_discharge(mask)
        # 시간에 따른 에너지 저장 및 방출을 나타냄

    def get_charge_up_schedule(self, mask):
    # DER (분산 에너지 자원)가 예약할 수 있는 상향 방향 (그리드로 전력을 제공하는 방향)의 충전 전력 양을 나타내는 것
        """ the amount of charging power in the up direction (supplying power up into the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}ZeroUp')

    def get_charge_down_schedule(self, mask):
    # DER (분산 에너지 자원)가 예약할 수 있는 하향 방향 (그리드에서 전력을 가져오는 방향)의 충전 전력 양을 나타내는 것
        """ the amount of charging power in the up direction (pulling power down from the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}ZeroDown')

    def get_discharge_up_schedule(self, mask):
    # DER (분산 에너지 자원)가 예약할 수 있는 상향 방향 (그리드로 전력을 제공하는 방향)의 방전 전력 양을 나타내는 것
        """ the amount of discharge power in the up direction (supplying power up into the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}ZeroUp')

    def get_discharge_down_schedule(self, mask):
    #  DER (분산 에너지 자원)가 예약할 수 있는 하향 방향 (그리드에서 전력을 가져오는 방향)의 방전 전력 양을 나타내는 것
        """ the amount of discharging power in the up direction (pulling power down from the grid) that
        this DER can schedule to reserve

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns: CVXPY parameter/variable

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}ZeroDown')

    def get_delta_uenegy(self, mask):
    # 현재 SOE (State of Energy) 수준에서 DER의 에너지 상태가 서브타임스텝 에너지 이동에 의해 변경되는 양을 나타내는 것
        """ the amount of energy, from the current SOE level the DER's state of energy changes
        from subtimestep energy shifting

        Returns: the energy throughput in kWh for this technology

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}Zero')

    def get_uenergy_increase(self, mask):
    # 분산 그리드에 제공되는 타임스텝 당 에너지 양을 나타내는 것
        """ the amount of energy in a timestep that is provided to the distribution grid

        Returns: the energy throughput in kWh for this technology

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}Zero')

    def get_uenergy_decrease(self, mask):
    # 분산 그리드에서 가져오는 타임스텝 당 에너지 양을 나타내는 것 
        """ the amount of energy in a timestep that is taken from the distribution grid

        Returns: the energy throughput in kWh for this technology

        """
        return cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name=f'{self.name}Zero')

    def objective_function(self, mask, annuity_scalar=1):
    # 김술과 관련된 목적 함수를 생성하는 것으로 보입니다. 기본적으로는 O&M(운영 및 유지) 비용을 포함할 수 있으며, 이는 0이 될 수 있임 
        """ Generates the objective function related to a technology. Default includes O&M which can be 0

        Args:
            mask (Series): Series of booleans used, the same length as case.power_kw
            annuity_scalar (float): a scalar value to be multiplied by any yearly cost or benefit that helps capture the cost/benefit over
                the entire project lifetime (only to be set iff sizing)

        Returns:
            costs - benefits (Dict): Dict of objective costs
        """
        return {} # 빈 딕셔너리, 반환 값은 목적 함수의 비용과 이익을 나타내는 딕셔너리 

    def constraints(self, mask, **kwargs):
    # 서비스에 제약 조건이 없는 경우에 사용되는 기본 제약 조건 목록 생성 메서드
        """Default build constraint list method. Used by services that do not have constraints.

        Args:
            mask (DataFrame): A boolean array that is true for indices corresponding to time_series data included
                    in the subs data set

        Returns:
            A list of constraints that corresponds the battery's physical constraints and its service constraints
        """
        return [] # 반환 값은 배터리의 물리적 제약 조건 및 서비스 제약 조건에 해당하는 제약 조건 목록

    def save_variable_results(self, subs_index):
    # 최적화 변수의 딕셔너리를 검색하고 각 DER 인스턴스에 특정한 변수 값을 자체에 저장하는 것
        """ Searches through the dictionary of optimization variables and saves the ones specific to each
        DER instance and saves the values it to itself

        Args:
            subs_index (Index): index of the subset of data for which the variables were solved for

        """
        variable_values = pd.DataFrame({name: variable.value for name, variable in self.variables_dict.items()}, index=subs_index)
        # 최적화 변수의 값을 DataFrame으로 저장
        # variables_dict 딕셔너리에서 각 변수의 이름과 값으로 이루어진 데이터프레임을 생성
        self.variables_df = pd.concat([self.variables_df, variable_values], sort=True)
        # self.variables_df에 새로 계산된 변수 값을 추가
        # 이전에 저장된 변수 값들과 함께 새로운 값을 결합
   # 최적화된 변수 값을 저장하고 이를 self.variables_df에 누적하여 기록

    def unique_tech_id(self):
    # 특정 DER에 대한 최적화 변수를 보고하는 데 사용되는 문자열 ID를 생성
        """ String id that serves as the prefix for reporting optimization variables for specific DER via timeseries
            or proforma method. USED IN REPORTING ONLY
        """
        return f'{self.tag.upper()}: {self.name}'
        # self.tag와 self.name을 조합하여 DER에 대한 고유한 문자열 ID를 생성

    def timeseries_report(self):
    # DER에 대한 최적화 결과를 요약하는 시계열 데이터프레임을 생성하는 것
        """ Summaries the optimization results for this DER.

        Returns: A timeseries dataframe with user-friendly column headers that summarize the results
            pertaining to this instance

        """
        # 반환 값은 이 DER 인스턴스에 관련된 결과를 요약하는 사용자 친화적인 열 헤더를 가진 시계열 데이터프레임
        return pd.DataFrame()

    def monthly_report(self):
    # 객체에 저장된 모든 월별 데이터를 수집하는 것
        """  Collects all monthly data that are saved within this object

        Returns: A dataframe with the monthly input price of the service

        """

    def drill_down_reports(self, monthly_data=None, time_series_data=None, technology_summary=None, sizing_df=None):
    # 사용자에게 보고되는 서비스 관련 데이터프레임을 계산
        """Calculates any service related dataframe that is reported to the user.

        Args:
            monthly_data:
            time_series_data:
            technology_summary:
            sizing_df:

        Returns: dictionary of DataFrames of any reports that are value stream specific
            keys are the file name that the df will be saved with
        """
        return {}
        # 반환 값은 값 스트림별로 보고되는 데이터프레임의 딕셔너리

    def proforma_report(self, apply_inflation_rate_func, fill_forward_func, results):
    # 값을 스트림에 참여하는 데 해당하는 프로포르마를 계산하는 것
        """ Calculates the proforma that corresponds to participation in this value stream

        Args:
            apply_inflation_rate_func:
            fill_forward_func:
            results (pd.DataFrame):

        Returns: A DateFrame of with each year in opt_year as the index and
            the corresponding value this stream provided.

        """
        if not self.zero_column_name(): # zero_column_name이 None이면 None을 반환
            return None
            # 제로 컬럼 이름이 설정되어 있지 않으면 계산할 수 없다는 것

        pro_forma = pd.DataFrame({self.zero_column_name(): -self.get_capex(solution=True)}, index=['CAPEX Year'])
        # zero_column_name을 사용하여 컬럼 이름을 가져온 다음, 해당 값 스트림의 CAPEX 값을 음수로 설정하여 데이터프레임을 생성
        # 인덱스는 'CAPEX Year'로 설정

        return pro_forma
        # 생성된 프로포르마 데이터프레임을 반환

    def interpolate_energy_dispatch(self, df, start_year, end_year, interpolation_method):
    # 누락된 값에 대해 주어진 방법으로 누락된 데이터를 보간하는 interpolate_energy_dispatch 함수
        """
        Interpolates cumulative energy dispatch values between
        the analysis start_year and end_year, given a dataframe with
        values only included for optimization years

        Args:
            df (pd.Dataframe): profroma type df with only years in the index,
                the years correspond to optimization years
            start_year (Integer): the project start year
            end_year (Integer): the project end year
            interpolation_method (String): defaults to 'linear'

        Returns: a df where the data is interpolated between known values using
            interpolation_method. Values prior to the first year with data get
            set to the first year's value. Same for the last year's value.
        """

        # default to linear interpolation
        if interpolation_method is None:
            interpolation_method = 'linear'
        # 보간 방법이 제공되지 않은 경우, 기본적으로 선형 보간을 사용

        filled_df = pd.DataFrame(index=pd.period_range(start_year, end_year, freq='y'))
        filled_df = pd.concat([filled_df, df], axis=1)
        # illed_df라는 새로운 데이터프레임을 생성하고, 인덱스로 start_year에서 end_year까지의 연도를 포함하도록 설정
        # 데이터프레임은 주어진 데이터프레임 df와 함께 합쳐짐

        filled_df = filled_df.apply(lambda x: x.interpolate(
            method=interpolation_method, limit_direction='both'), axis=0)
        # interpolate 메서드를 사용하여 보간을 수행
        # limit_direction='both'는 누락된 값의 양쪽 방향으로 보간을 의미, axis=0는 열 방향으로 보간

        return filled_df
        # 보간된 데이터프레임을 반환
        # 보간되지 않은 값은 각 열의 첫 번째 연도의 값으로 설정, 마지막 연도에 대해서도 동일하게 적용
