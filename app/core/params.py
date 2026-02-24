# Parameter schema and defaults for each mesh method

MESH_METHODS = {
    'pyvista': {
        'label': 'PyVista Delaunay 3D',
        'params': {
            'alpha': {
                'type': 'number',
                'default': 2.0,
                'min': 0.1,
                'max': 10.0,
                'step': 0.1,
                'description': 'Alpha值控制网格密度：值越小网格越紧密，越大则越松散。建议范围1-5'
            },
            'smooth_iter': {
                'type': 'number',
                'default': 1000,
                'min': 0,
                'max': 5000,
                'step': 100,
                'description': '平滑迭代次数：值越大网格表面越光滑，但可能丢失细节。推荐500-2000'
            }
        }
    },
    'alphashape': {
        'label': 'Alpha Shape',
        'params': {
            'alpha': {
                'type': 'number',
                'default': 2.0,
                'min': 0.1,
                'max': 10.0,
                'step': 0.1,
                'description': 'Alpha值决定形状包络程度：小值产生紧致外壳，大值产生凸包。适用于凹形物体'
            },
            'smooth_iter': {
                'type': 'number',
                'default': 500,
                'min': 0,
                'max': 5000,
                'step': 100,
                'description': '拉普拉斯平滑迭代次数：增加可消除噪声和小凸起。推荐300-1000'
            }
        }
    },
    'ballpivoting': {
        'label': 'Ball Pivoting',
        'params': {
            'radii': {
                'type': 'text',
                'default': '1.0,2.0,3.0',
                'description': '球体半径序列（逗号分隔）：从小到大填充不同尺度的孔洞。如：0.5,1.0,2.0'
            },
            'smooth_iter': {
                'type': 'number',
                'default': 500,
                'min': 0,
                'max': 5000,
                'step': 100,
                'description': '平滑次数：减少表面凹凸不平。对点云噪声大的数据建议1000+'
            }
        }
    },
    'poisson': {
        'label': 'Poisson Surface Reconstruction',
        'params': {
            'depth': {
                'type': 'number',
                'default': 8,
                'min': 1,
                'max': 14,
                'step': 1,
                'description': '八叉树深度：控制重建精度，越大越精细但计算慢。小数据用6-8，大数据用9-11'
            },
            'width': {
                'type': 'number',
                'default': 0,
                'min': 0,
                'max': 10,
                'step': 1,
                'description': '八叉树宽度：0表示自动。手动设置可控制最细层级宽度'
            },
            'scale': {
                'type': 'number',
                'default': 1.1,
                'min': 1.0,
                'max': 2.0,
                'step': 0.1,
                'description': '缩放因子：重建立方体与数据包络盒的比例。1.1-1.3适合大多数情况'
            },
            'linear_fit': {
                'type': 'boolean',
                'default': False,
                'description': '线性插值：启用可得到更平滑结果，但可能丢失尖锐特征'
            },
            'density_threshold': {
                'type': 'number',
                'default': 0.01,
                'min': 0.0,
                'max': 1.0,
                'step': 0.01,
                'description': '密度阈值百分位数：移除低密度顶点（噪声）。0.01-0.05可去除大部分异常点'
            },
            'smooth_iter': {
                'type': 'number',
                'default': 500,
                'min': 0,
                'max': 5000,
                'step': 100,
                'description': '后处理平滑次数：Poisson重建后的额外平滑。通常较小值即可（300-800）'
            }
        }
    },
    'marchingcube': {
        'label': 'Marching Cubes',
        'params': {
            'levelset': {
                'type': 'number',
                'default': 0,
                'min': -1,
                'max': 1,
                'step': 0.1,
                'description': '等值面阈值：0表示在体素边界提取，调整可获得膨胀/收缩效果'
            },
            'mc_scale_factor': {
                'type': 'number',
                'default': 1.0,
                'min': 0.1,
                'max': 5.0,
                'step': 0.1,
                'description': '体素缩放因子：影响网格分辨率，越大越粗糙但速度快。1.0-2.0通常最佳'
            },
            'dist_sample_num': {
                'type': 'number',
                'default': 100,
                'min': 10,
                'max': 1000,
                'step': 10,
                'description': '距离采样数：用于计算点云密度的样本数。100-200适合中小数据集'
            },
            'smooth_iter': {
                'type': 'number',
                'default': 500,
                'min': 0,
                'max': 5000,
                'step': 100,
                'description': '平滑迭代：Marching Cubes易产生阶梯，需较多平滑。建议800-2000'
            }
        }
    },
    'voxel': {
        'label': 'Voxel (Density + Contour)',
        'params': {
            'grid_size': {
                'type': 'number',
                'default': 60,
                'min': 20,
                'max': 200,
                'step': 5,
                'description': '体素网格分辨率：越大细节越多但更慢/更占内存。建议40-120'
            },
            'sigma': {
                'type': 'number',
                'default': 2.5,
                'min': 0.2,
                'max': 8.0,
                'step': 0.1,
                'description': '高斯平滑强度：越大越“组织感/圆润”，越小越贴近点云。建议1.0-3.5'
            },
            'iso_percentile': {
                'type': 'number',
                'default': 70,
                'min': 50,
                'max': 95,
                'step': 1,
                'description': '等值面阈值（密度分位数）：值越大外壳越“紧/瘦”，值越小越“胖/膨胀”。建议65-80'
            },
            'hole_size_factor': {
                'type': 'number',
                'default': 0.6,
                'min': 0.0,
                'max': 1.0,
                'step': 0.05,
                'description': '封洞强度系数：按模型包围盒对角线比例计算最大孔洞直径。越大越容易把首尾封上（也更可能过封）。建议0.2-0.6'
            },
            'smooth_iter': {
                'type': 'number',
                'default': 30,
                'min': 0,
                'max': 500,
                'step': 5,
                'description': '等值面后的平滑迭代次数：用于让封洞“盖子”过渡自然。建议10-60'
            },
            'closure_smooth_factor': {
                'type': 'number',
                'default': 1.0,
                'min': 0.5,
                'max': 3.0,
                'step': 0.1,
                'description': '封洞平整强度：越大越平滑（会更圆润且更慢），越小保留更多局部形状。建议0.8-1.8'
            }
        }
    }
}


def get_method_params(method):
    """Get parameter configuration for a specific method"""
    return MESH_METHODS.get(method, {}).get('params', {})


def get_default_params(method):
    """Get default parameter values for a specific method"""
    params = get_method_params(method)
    return {key: config['default'] for key, config in params.items()}


def parse_radii(radii_str):
    """Parse comma-separated radii string into list of floats"""
    try:
        return [float(r.strip()) for r in radii_str.split(',') if r.strip()]
    except ValueError:
        raise ValueError(f"Invalid radii format: {radii_str}. Expected comma-separated numbers.")
