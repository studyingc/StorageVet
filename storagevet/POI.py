"""POI.py

"""
import numpy as np
import cvxpy as cvx
import pandas as pd
from storagevet.ErrorHandling import *


class POI:
    # POI: point of interconenection
    # 에너지 시스템의 포인트 오브 인터커넥션을 나타내는 클래스
    # 부하 데이터 저장, POI에서 적용해야할 제약 조건 부과

    def __init__(self, params, technology_inputs_map, technology_class_map):
        """ POI 객체를 초기화하하는 함수
        사용자가 정의한 모델 매개변수 및 기술 입력을 사용하여 DER 객체들을 초기화하고 활성 DER 목록에 추가함
        """
        self.max_export = params['max_export']  # > 0
        self.max_import = params['max_import']  # < 0
        self.apply_poi_constraints = params['apply_poi_constraints']

        # types of DER technology, and their dictionary of active specific technologies
        self.der_list = []
        self.der_summary = {}  # keys= names of DERs, values= DER type  (basically tech summary output)
        self.active_ders = []

        # initialize all DERs
        for der, params_input in technology_inputs_map.items():
            if params_input is not None:  # then Params class found an input
                for id_val, id_der_input in params_input.items():
                    # add ID to der input dictionary
                    id_der_input['ID'] = id_val
                    TellUser.info(f"Initializing: {der}-{id_val}")
                    der_object = technology_class_map[der](id_der_input)
                    self.der_list.append(der_object)
                    self.active_ders.append(der_object)

    def calculate_system_size(self):
        """ 시스템이 추출할 수 있는 최대 전력, 수입할 수 있는 최대 전력, 저장할 수 있는 최대 에너지 및 최소 에너지를 계산함
        """
        min_ene = 0
        max_ene = 0
        max_ch = 0
        max_dis = 0

        for der_instance in self.active_ders: # 활성화된 DER 사례들을 순회
            min_ene += der_instance.operational_min_energy()
            max_ene += der_instance.operational_max_energy()
            max_ch += der_instance.charge_capacity()
            max_dis += der_instance.discharge_capacity()
        return max_ch, max_dis, (max_ene, min_ene) # 시스템의 크기를 계산하고 반환

    def initialize_optimization_variables(self, size):
        """ 최적화 루프의 시작에서 최적화 변수를 초기화하는 함수 
        """
        for der_instance in self.active_ders:
            # initialize variables
            der_instance.initialize_variables(size)

    def get_state_of_system(self, mask):
        """ 
        POI의 상태를 측정하여 최적화에 사용될 파라미터를 계산하는 함수
        SET_UP_OPTIMIZATION에서 호출됨

        반환값
        - load sum: 부하 집계
        - var_gen_sum: 가변 자원으로부터의 발전
        - gen_sum: 다른 출처에서의 발전
        - tot_net_ess: ESS의 총 순 전력
        - der_dispatch_net_power: 디스패처블 DERs(중단되지 않는 자원 및 부하가 아님)로부터의 순 전력
        - total_soe: 시스템에 저장된 에너지의 총 상태
        - agg_power_flows_in: POI로의 모든 전력 흐름 집계
        - agg_power_flows_out: POI에서의 모든 전력 흐름 집계
        
        - agg_steam_heating_power: 스팀 열화력(열 회수된 열)의 집계
        - agg_hotwater_heating_power: 핫워터 열화력(열 회수된 열)의 집계
        - agg_thermal_cooling_power: 열 냉각력(냉각 회수된 열)의 집계
        """
        opt_var_size = sum(mask)
        load_sum = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')  # at POI
        var_gen_sum = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')  # at POI
        gen_sum = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')
        tot_net_ess = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')
        der_dispatch_net_power = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')
        total_soe = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')
        agg_power_flows_in = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')  # at POI
        agg_power_flows_out = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')  # at POI

        agg_steam_heating_power = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')  # at POI
        agg_hotwater_heating_power = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')  # at POI
        agg_thermal_cooling_power = cvx.Parameter(value=np.zeros(opt_var_size), shape=opt_var_size, name='POI-Zero')  # at POI
        # 최적화 위한 변수 설정
        
        for der_instance in self.active_ders:
            # add the state of the der's power over time & stored energy over time to system's
            # 시간에 따른 전력, 에너지 저장 상태를 시스템의 변수에 추가
            # these agg_power variables are used with POI constraints
            agg_power_flows_in += (der_instance.get_charge(mask) - der_instance.get_discharge(mask))
            agg_power_flows_out += (der_instance.get_discharge(mask) - der_instance.get_charge(mask))
            # 전력 흐름 계산

            if der_instance.technology_type == 'Load': # 기술 유형 확인
                load_sum += der_instance.get_charge(mask)
                # 해당 DER의 충전 값을 load_sum 변수에 추가
            if der_instance.technology_type == 'Energy Storage System':
                total_soe += der_instance.get_state_of_energy(mask)
                tot_net_ess += der_instance.get_net_power(mask)
                # auxiliary load (hp) for a Battery will contribute to agg_power variables 
                # auxiliary load: 보조 부하
                #   in a similar manner to SiteLoad (add to flow_in and subtract from flow_out)
                try: # 배터리에 보조 부하가 잇는 경우
                    aux_load = der_instance.hp
                    agg_power_flows_in += aux_load # 해당 보조부하를 추가
                    agg_power_flows_out -= aux_load # 해당 보조부하를 뺌
                except AttributeError:
                    pass
            if der_instance.technology_type == 'Generator':
                gen_sum += der_instance.get_discharge(mask)
            if der_instance.technology_type == 'Intermittent Resource':
                var_gen_sum += der_instance.get_discharge(mask)
            if der_instance.technology_type in ['Energy Storage System', 'Generator']:
                der_dispatch_net_power += der_instance.get_net_power(mask)

        ##NOTE: these print statements disclose info for get_state_of_system cvx Parameters
        #print(f'load_sum:               ({load_sum})')
        #print(f'var_gen_sum:            ({var_gen_sum})')
        #print(f'gen_sum:                ({gen_sum})')
        #print(f'tot_net_ess:            ({tot_net_ess})')
        #print(f'der_dispatch_net_power: ({der_dispatch_net_power})')
        #print(f'total_soe:              ({total_soe})')
        #print(f'agg_power_flows_in:     ({agg_power_flows_in.name()})')
        #print(f'agg_power_flows_out:    ({agg_power_flows_out.name()})')

        return load_sum, var_gen_sum, gen_sum, tot_net_ess, der_dispatch_net_power, total_soe, agg_power_flows_in, agg_power_flows_out, agg_steam_heating_power, agg_hotwater_heating_power, agg_thermal_cooling_power

    def combined_discharge_rating_for_reliability(self):
        # 신뢰성 전력 제약 조건을 생성
        
        combined_rating = 0
        for der_instance in self.active_ders:
            if der_instance.technology_type == 'Energy Storage System': #
                combined_rating += der_instance.dis_max_rated # ESS의 최대 방전 평가를 추가
            if der_instance.technology_type == 'Generator':
                combined_rating += der_instance.rated_power # 발전기의 평가된 전력을 추가
        return combined_rating
        # ESS과 ICE(발전기)의 결합 방전 등급을 얻을 수 있음

    def optimization_problem(self, mask, power_in, power_out, steam_in, hotwater_in, cold_in, annuity_scalar=1):
        """ 
        최적화 문제를 생성하고 POI의 제약 조건 리스트를 반환하는 함수

        Power_in: 전력 입력
        Power_out: 전력 출력
        Steam_in: 스팀 입력
        hotwater_in: 핫워터 입력
        cold_in: 냉수 입력
        annuity_scalr(float): 전체 프로젝트 수명 동안 비용/수익을 포착하는 데 도움이 되는 연간 비용 또는 이익에 곱해지는 스칼라 값

        반환값
        - 영향을 받는 목적 함수의 일부분을 나타내는 표현식의 키로 라벨이 지정된 딕셔너리 
        - POI에 의해 설정된 제약 조건 목록: 전력 예약, 제어 제약 조건 요구 사항, 최대 수입, 최대 수출 등
        """
        constraint_list = [] 
        opt_size = sum(mask) # 최적화에 사용될 크기
        obj_expression = {}  # dict of objective costs

        # deal with grid_charge constraint btw ESS and PV
        total_pv_out_ess_can_charge_from = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POIZero')
        total_ess_charge = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POIZero')
        allow_charge_from_grid = True
        # deal with inverter constraint btw ess and any PV marked as dc coupled
        total_pv_out_dc = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POIZero')
        net_ess_power = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POIZero')
        agg_inv_max = 0
        dc_coupled_pvs = False

        # total_pv_out_ess_can_charge_from: 에너지 저장 시스템이 충전할 수 잇는 PV(태양광)의 총 출력
        # total_ess_charge: 에너지 저장 시스템의 총 충전량
        # allow_chage_from_grid: 그리드에서 충전을 허용하는지 여부를 결정하는 불리언 값
        # total_pv_out_dc: 직류 결합된 PV와 ESS 사이의 인버터 제약 조건을 처리하기 위한 pv의 총출력 / PV의 DC 출력
        # net_ess_power: ESS의 순 전력
        # agg_inv_max: 집계된 인버터의 최대값
        # dc_coupled_pvs: 직류 결합된 PV 시스템이 있는지 여부를 나타내는 불리언 값

        for der_instance in self.active_ders:
            # add all operational constraints
            constraint_list += der_instance.constraints(mask)
            # 운용 제약 조건을 계산하고 제약 조건 리스트에 추가
            # add DER cost funcs
            obj_expression.update(der_instance.objective_function(mask, annuity_scalar))
            # 목적 함수를 계산하고 목적 함수 표현식 딕셔너리를 업데이트
            if der_instance.tag == 'PV': # DER의 태그가 'PV'인 경우
                if not der_instance.grid_charge: # 그리드 충전을 허용하지 않는 경우
                    allow_charge_from_grid = False
                    total_pv_out_ess_can_charge_from += der_instance.get_discharge(mask) # DER의 방전 값을 추가
                if der_instance.loc == 'dc': # 위치가 'dc'인 경우
                    dc_coupled_pvs = True
                    total_pv_out_dc += der_instance.get_discharge(mask) # DER의 방전 값을 추가
                    agg_inv_max += der_instance.inv_max # DER의 인버터 최대값을 추가
            if der_instance.technology_type == 'Energy Storage System': # 기술 유형이 'Energy Storage System'인 경우
                net_ess_power += der_instance.get_net_power(mask) # DER의 순 전력을 추가
                total_ess_charge += der_instance.get_charge(mask) # DER의 충전 값을 추가

        if not allow_charge_from_grid:  # 그리드 충전을 허용하지 않는 경우, add grid charge constraint
            constraint_list += [cvx.NonPos(total_ess_charge - total_pv_out_ess_can_charge_from)]
            # 에너지 저장 시스템(ESS)이 태양광(PV)으로부터만 충전되는 경우에 해당

        if dc_coupled_pvs:  # 직류 결합된 PV 시스템이 있는 경우, add dc coupling constraints
            constraint_list += [cvx.NonPos(total_pv_out_dc + (-1) * net_ess_power - agg_inv_max)]
            constraint_list += [cvx.NonPos(-agg_inv_max - total_pv_out_dc + net_ess_power)]
            # 직류 결합된 PV 시스템과 ESS 사이의 인버터 제약 조건이 최적화 문제에 추가되어 올바른 동작을 보장

        # power import/export constraints
        # NOTE: power_in is agg_p_in (defined in self.get_state_of_system)
        # NOTE: power_out is agg_p_out (defined in self.get_state_of_system)
        if self.apply_poi_constraints: # POI 제약 조건을 적용하는 경우
            # (agg_p_in) <= -max_import, 총 전력 수입이 최대 수입량 이하임을 보장
            constraint_list += [cvx.NonPos(self.max_import + power_in)]

            # (agg_p_out) <= max_export, 총 전력 수출이 최대 수출량 이하임을 보장
            # NOTE: with active_load_dump True, we do not include this constraint
            #   active_load_dump only occurs in DER-VET, DER-VET에서만 발생하며, 이 경우 최대 수출 제약 조건을 비활성화
            if not self.disable_max_export_poi_constraint():
                constraint_list += [cvx.NonPos(power_out + -1 * self.max_export)]
            # 최대 수입 및 최대 수출 제약 조건이 최적화 문제에 추가되어 올바른 동작을 보장

        return obj_expression, constraint_list
        # 목적 함수 표현식과 제약 조건 리스트를 반환

    def disable_max_export_poi_constraint(self):
        # NOTE: active_load_dump only occurs in DER-VET, DER-VET에서만 사용가능 ==> 그럼 굳이 사용할 필요가 없는 코드인가?
        # 최대 수출 POI 제약을 비활성화하는지 여부를 반환하는 함수수
        try:
            disable_max_export = self.active_load_dump # active_load_dump의 값을 변수에 할당
            if disable_max_export: # active_load_dump가 True인 경우
                TellUser.warning(f'You have activated a load dump in DER-VET, therefore we disable the max_export POI constraint in the optimization. The load dump will be determined as a post-optimization calculation.')
            return disable_max_export
        except AttributeError: # active_load_dump 속성이 정의되지 않은 경우
            return False # 최대 수출 POI 제약 조건이 비활성화되지 않음

    def aggregate_p_schedules(self, mask):
        """ 
        DER(분산 에너지 자원)의 방전 전력 일정을 합산하는 기능을 수행
        '전력 일정(Power Schedule)'은 기술의 방전 용량이 전력을 더 많이 공급하도록 입찰할 수 있는 양(UP) 또는 전력을 더 적게 공급하도록 입찰할 수 있는 양(DOWN)을 정의
        POI가 연결된 전기 그리드에서 이루어지는 것
        """
        opt_size = sum(mask)
        agg_dis_up = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POI-Zero')
        agg_dis_down = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POI-Zero')
        agg_ch_up = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POI-Zero')
        agg_ch_down = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POI-Zero')
        uenergy_incr = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POI-Zero')
        uenergy_decr = cvx.Parameter(value=np.zeros(opt_size), shape=opt_size, name='POI-Zero')
        uenergy_thru = cvx.Parameter(value=np.zeros(sum(mask)), shape=sum(mask), name='ServiceAggZero')

        # agg_dis_up: 전력을 더 많이 공급
        # agg_dis_down: 역방향으로 전력을 끌어내림
        # agg_ch_up: 그리드로부터 더 많은 전력을 받아들임
        # agg_ch_down: 역방향으로 그리드로 전력을 밀어올림
        # uenergy_incr: 서브 타임 스텝 활동 중에 저장된 에너지를 증가
        # uenergy_decr: 서브 타임 스텝 활동 중에 저장된 에너지를 감소
        # uenergy_thru: 서브 타임 스텝 활동 중 전달된 총 에너지
        
        for der_in_market_participation in self.active_ders: # active_ders 리스트에 있는 각 DER에 대해 반복
            if der_in_market_participation.can_participate_in_market_services: # DER이 시장 서비스에 참여할 수 있는지 확인
                agg_ch_up += der_in_market_participation.get_charge_up_schedule(mask) # DER의 충전 UP 일정 누적
                agg_ch_down += der_in_market_participation.get_charge_down_schedule(mask) # DER의 충전 DOWN 일정 누적 
                agg_dis_up += der_in_market_participation.get_discharge_up_schedule(mask) # DER의 방전 UP 일정 누적
                agg_dis_down += der_in_market_participation.get_discharge_down_schedule(mask) # DER의 방전 DOWN 일정 누적
                uenergy_incr += der_in_market_participation.get_uenergy_increase(mask) # DER의 에너지 증가 누적 
                uenergy_decr += der_in_market_participation.get_uenergy_decrease(mask) # DER의 에너지 감소 누적
                uenergy_thru += der_in_market_participation.get_delta_uenegy(mask) # DER의 에너지 변경 누적

        return agg_dis_down, agg_dis_up, agg_ch_down, agg_ch_up, uenergy_decr, uenergy_incr, uenergy_thru
        # 누적된 값들 반환

    def merge_reports(self, is_dispatch_opt, index):
        """ DER들의 최적화 결과를 수집하고 병합하여 사용자 친화적인 결과를 나타내는 데이터프레임을 반환하는 함수수
        """
        results = pd.DataFrame(index=index)
        monthly_data = pd.DataFrame()

        # initialize all the data columns that will ALWAYS be present in our results
        results.loc[:, 'Total Load (kW)'] = 0
        results.loc[:, 'Total Generation (kW)'] = 0
        results.loc[:, 'Total Storage Power (kW)'] = 0
        results.loc[:, 'Aggregated State of Energy (kWh)'] = 0

        for der_instance in self.der_list:
            report_df = der_instance.timeseries_report()
            results = pd.concat([report_df, results], axis=1)
            if der_instance.technology_type in ['Generator', 'Intermittent Resource']:
                results.loc[:, 'Total Generation (kW)'] += results[f'{der_instance.unique_tech_id()} Electric Generation (kW)']
            if der_instance.technology_type == 'Energy Storage System':
                results.loc[:, 'Total Storage Power (kW)'] += results[f'{der_instance.unique_tech_id()} Power (kW)']
                results.loc[:, 'Aggregated State of Energy (kWh)'] += results[f'{der_instance.unique_tech_id()} State of Energy (kWh)']
                # add any battery auxiliary load (hp) to the total load
                try:
                    auxiliary_load = der_instance.hp
                    if auxiliary_load > 0:
                        TellUser.info(f'adding auxiliary load ({auxiliary_load} from ESS: {der_instance.name} to the Total Load')
                        results.loc[:, 'Total Load (kW)'] += auxiliary_load
                except AttributeError:
                    pass
            if der_instance.technology_type == 'Load':
                results.loc[:, 'Total Load (kW)'] += results[f'{der_instance.unique_tech_id()} Original Load (kW)']
            report = der_instance.monthly_report()
            monthly_data = pd.concat([monthly_data, report], axis=1, sort=False)

        # assumes the orginal net load only does not contain the Storage system
        # net load is the load see at the POI
        results.loc[:, 'Net Load (kW)'] = results.loc[:, 'Total Load (kW)'] - results.loc[:, 'Total Generation (kW)'] - results.loc[:, 'Total Storage Power (kW)']
        return results, monthly_data

    def technology_summary(self):
        """ DER들의 종류 및 이름에 대한 데이터프레임을 생성하여 반환하는 함수
        """
        der_summary = {'Type': [], 'Name': []}
        for der_object in self.der_list:
            der_summary['Type'].append(der_object.technology_type)
            der_summary['Name'].append(der_object.name)
        technology_summary = pd.DataFrame(der_summary)
        technology_summary.set_index('Name')
        return technology_summary

    def drill_down_dfs(self, **kwargs):
        """ 결과를 세부적으로 분석하기 위한 데이터프레임을 반환하는 함수
        """
        df_dict = dict()
        for der in self.der_list:
            df_dict.update(der.drill_down_reports(**kwargs))
        return df_dict
