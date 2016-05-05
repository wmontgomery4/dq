#include <boost/numpy.hpp>
#include <cmath>
#include "macros.h"
#include <iostream>
#include <boost/python/slice.hpp>
#include "mujoco_osg_viewer.hpp"

namespace bp = boost::python;
namespace bn = boost::numpy;




namespace {

bp::object main_namespace;

template<typename T>
bn::ndarray toNdarray1(const T* data, long dim0) {
  long dims[1] = {dim0};
  bn::ndarray out = bn::empty(1, dims, bn::dtype::get_builtin<T>());
  memcpy(out.get_data(), data, dim0*sizeof(T));
  return out;
}
template<typename T>
bn::ndarray toNdarray2(const T* data, long dim0, long dim1) {
  long dims[2] = {dim0,dim1};
  bn::ndarray out = bn::empty(2, dims, bn::dtype::get_builtin<T>());
  memcpy(out.get_data(), data, dim0*dim1*sizeof(T));
  return out;
}
template<typename T>
bn::ndarray toNdarray3(const T* data, long dim0, long dim1, long dim2) {
  long dims[3] = {dim0,dim1,dim2};
  bn::ndarray out = bn::empty(3, dims, bn::dtype::get_builtin<T>());
  memcpy(out.get_data(), data, dim0*dim1*dim2*sizeof(T));
  return out;
}


bool endswith(const std::string& fullString, const std::string& ending)
{
	return (fullString.length() >= ending.length()) && 
		(0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
}




class PyMJCWorld2 {


public:

    PyMJCWorld2(const std::string& loadfile);
    bp::object Step(const bn::ndarray& x, const bn::ndarray& u);
    bn::ndarray PackObs(const bp::dict fields, const bp::list xyz, const bp::object dO);
    void Plot(const bn::ndarray& x);    
    void Idle(const bn::ndarray& x);
    bn::ndarray GetCOMMulti(const bn::ndarray& x);
    bn::ndarray GetJacSite(int site);
    void Kinematics();
    bp::dict GetModel();
    void SetModel(bp::dict d);
    bp::dict GetData();
    void SetData(bp::dict d);
    bn::ndarray GetImage(const bn::ndarray& x);
    void SetNumSteps(int n) {m_numSteps=n;}

    ~PyMJCWorld2();
private:
    // PyMJCWorld(const PyMJCWorld&) {}

    void _PlotInit();

    // void _SetState(const mjtNum* xdata) {mju_copy(m_data->qpos, (xdata), NQ); mju_copy(m_data->qvel, (xdata)+NQ, NV); }
    // void _SetControl(const mjtNum* udata) {for (int i=0; i < m_actuatedDims.size(); ++i) m_u[m_actuatedDims[i]] = (udata)[i];}


    mjModel* m_model;
    mjData* m_data;
    MujocoOSGViewer* m_viewer;
    int m_numSteps;
    int m_featmask;

};

PyMJCWorld2::PyMJCWorld2(const std::string& loadfile) {
    mj_activate("src/3rdparty/mjpro/mjkey.txt");
  	if (endswith(loadfile, "xml")) {
        NewModelFromXML(loadfile.c_str(), m_model);
  	}
  	else {
  	    NOTIMPLEMENTED;
  	}	
    if (!m_model) PRINT_AND_THROW("couldn't load model: " + std::string(loadfile));
    m_data = mj_makeData(m_model);
    FAIL_IF_FALSE(!!m_data);
    m_viewer = NULL;
    m_numSteps = 1;
    m_featmask = 0;
}


PyMJCWorld2::~PyMJCWorld2() {
	if (m_viewer) {
		delete m_viewer;
	}
	mj_deleteData(m_data);
	mj_deleteModel(m_model);
}


int StateSize(mjModel* m) {
    return m->nq + m->nv;
}
void GetState(mjtNum* ptr, const mjModel* m, const mjData* d) {
    mju_copy(ptr, d->qpos, m->nq);
    ptr += m->nq;
    mju_copy(ptr, d->qvel, m->nv);
}
void SetState(const mjtNum* ptr, const mjModel* m, mjData* d) {
    mju_copy(d->qpos, ptr, m->nq);
    ptr += m->nq;
    mju_copy(d->qvel, ptr, m->nv);
}
inline void SetCtrl(const mjtNum* ptr, const mjModel* m, mjData* d) {
    mju_copy(d->ctrl, ptr, m->nu);
}

#define MJTNUM_DTYPE bn::dtype::get_builtin<mjtNum>()

bp::object PyMJCWorld2::Step(const bn::ndarray& x, const bn::ndarray& u) {
    FAIL_IF_FALSE(x.get_dtype() == MJTNUM_DTYPE && x.get_nd() == 1 && x.get_flags() & bn::ndarray::C_CONTIGUOUS && x.shape(0) == m_model->nq+m_model->nv);
    FAIL_IF_FALSE(u.get_dtype() == MJTNUM_DTYPE && u.get_nd() == 1 && u.get_flags() & bn::ndarray::C_CONTIGUOUS && u.shape(0) == m_model->nu);

    SetState(reinterpret_cast<const mjtNum*>(x.get_data()), m_model, m_data);

    mj_step1(m_model,m_data);
    SetCtrl(reinterpret_cast<const mjtNum*>(u.get_data()), m_model, m_data);
    mj_step2(m_model,m_data);

    // mj_kinematics(m_model, m_data);
    // printf("before: %f\n", m_data->com[0]);
/*
    for (int i=0; i < m_numSteps; ++i) mj_step(m_model,m_data);
    if (m_featmask & feat_cfrc_ext) mj_rnePost(m_model, m_data);
*/
    // printf("after step: %f\n", m_data->com[0]);
    // mj_kinematics(m_model, m_data);
    // printf("after step+kin: %f\n", m_data->com[0]);

    long xdims[1] = {StateSize(m_model)};
    long site_dims[2] = {m_model->nsite, 3};
    bn::ndarray xout = bn::empty(1, xdims, bn::dtype::get_builtin<mjtNum>());
    bn::ndarray site_out = bn::empty(2, site_dims, bn::dtype::get_builtin<mjtNum>());

    GetState((mjtNum*)xout.get_data(), m_model, m_data);
    mju_copy((mjtNum*)site_out.get_data(), m_data->site_xpos, 3*m_model->nsite);

	return bp::make_tuple(xout, site_out);
}

/*
 * Pack an observation given the desired fields and arguments
 * TODO: write out this documentation
 */
bn::ndarray PyMJCWorld2::PackObs(const bp::dict fields, const bp::list xyz, const bp::object dO) {
    // Create output
    int odims = bp::extract<int>(dO);
    long shape[1] = {odims};
    bn::ndarray obs = bn::empty(1, shape, bn::dtype::get_builtin<mjtNum>());

    // Convert python list to array
    bool mask[3];
    for (int i=0; i < 3; i++){
        mask[i] = bp::extract<bool>(xyz[i]);
    }

    // Iterate over possible fields
    mjtNum* ptr = (mjtNum*) obs.get_data();
    int idx = 0;
    if (fields.has_key("qpos")){
        mju_copy(ptr+idx, m_data->qpos, m_model->nq);
        idx += m_model->nq;
    }
    if (fields.has_key("qvel")){
        mju_copy(ptr+idx, m_data->qvel, m_model->nv);
        idx += m_model->nv;
    }
    if (fields.has_key("xipos")){
        // Skip worldbody
        mjtNum* xipos_ptr = m_data->xipos + 3;
        for (int i = 1; i < m_model->nbody; i++){
            for (int j = 0; j < 3; j++){
                if (mask[j]){
                    mju_copy(ptr+idx, xipos_ptr+j, 1);
                    idx += 1;
                }
            }
            xipos_ptr += 3;
        }
    }
    if (fields.has_key("ximat")){
        // Skip worldbody
        mjtNum* ximat_ptr = m_data->ximat + 9;
        for (int i = 1; i < m_model->nbody; i++){
            for (int j = 0; j < 3; j++){
                for (int k = 0; k < 3; k++){
                    if (mask[j] && mask[k]){
                        mju_copy(ptr+idx, ximat_ptr+3*i+j, 1);
                        idx += 1;
                    }
                }
            }
            ximat_ptr += 9;
        }
    }
    if (fields.has_key("site_xpos")){
        mjtNum* site_ptr = m_data->site_xpos;
        for (int i = 0; i < m_model->nsite; i++){
            for (int j = 0; j < 3; j++){
                if (mask[j]){
                    mju_copy(ptr+idx, site_ptr+j, 1);
                    idx += 1;
                }
            }
            site_ptr += 3;
        }
    }
    if (fields.has_key("to_target")){
        // Only works for end effector and target
        // Note that ordering of end effector/target doesn't matter
        FAIL_IF_FALSE(m_model->nsite == 2);
        mjtNum* s0 = m_data->site_xpos;
        mjtNum* s1 = s0 + 3;

        // Add in differences along observed dimensions
        for (int i = 0; i < 3; i++){
            if (mask[i]){
                mju_addScl(ptr+idx, s0+i, s1+i, -1.0, 1);
                idx += 1;
            }
        }
    }

    // Check that proper dO was given
    FAIL_IF_FALSE(idx == dO);
    return obs;
}


void GetCOM(const mjModel* m, const mjData* d, mjtNum* com) {
    // see mj_com in engine_core.c
    mjtNum tot=0;
    com[0] = com[1] = com[2] = 0;
    for(int i=1; i<m->nbody; i++ ) {
        com[0] += d->xipos[3*i+0]*m->body_mass[i];
        com[1] += d->xipos[3*i+1]*m->body_mass[i];
        com[2] += d->xipos[3*i+2]*m->body_mass[i];
        tot += m->body_mass[i];
    }
    // compute com
    com[0] /= tot;
    com[1] /= tot;
    com[2] /= tot;
}

bn::ndarray PyMJCWorld2::GetCOMMulti(const bn::ndarray& x) {
    int state_size = StateSize(m_model);
    FAIL_IF_FALSE(x.get_dtype() == MJTNUM_DTYPE && x.get_nd() == 2 && x.get_flags() & bn::ndarray::C_CONTIGUOUS && x.shape(1) == state_size);
    int N = x.shape(0);
    long outdims[2] = {N,3};
    bn::ndarray out = bn::empty(2, outdims, bn::dtype::get_builtin<mjtNum>());
    mjtNum* ptr = (mjtNum*)out.get_data();
    for (int n=0; n < N; ++n) {
        SetState(reinterpret_cast<const mjtNum*>(x.get_data()), m_model, m_data);
        mj_kinematics(m_model, m_data);
        GetCOM(m_model, m_data, ptr);
        ptr += 3;
    }
    return out;
}

bn::ndarray PyMJCWorld2::GetJacSite(int site) {
    bn::ndarray out = bn::zeros(bp::make_tuple(3,m_model->nv), bn::dtype::get_builtin<mjtNum>());
    mjtNum* ptr = (mjtNum*)out.get_data();
    mj_jacSite(m_model, m_data, ptr, 0, site);
    return out;
}

void PyMJCWorld2::Kinematics() {
    mj_kinematics(m_model, m_data);
    mj_comPos(m_model, m_data);
    mj_tendon(m_model, m_data);
    mj_transmission(m_model, m_data);
}

void PyMJCWorld2::_PlotInit() {
    if (m_viewer == NULL) {
        m_viewer = new MujocoOSGViewer();
        m_viewer->SetModel(m_model);
    }
}

void PyMJCWorld2::Plot(const bn::ndarray& x) {
    FAIL_IF_FALSE(x.get_dtype() == MJTNUM_DTYPE && x.get_nd() == 1 && x.get_flags() & bn::ndarray::C_CONTIGUOUS);
    _PlotInit();
    SetState(reinterpret_cast<const mjtNum*>(x.get_data()),m_model,m_data);
	m_viewer->SetData(m_data);
	m_viewer->RenderOnce();
}

void PyMJCWorld2::Idle(const bn::ndarray& x) {
    FAIL_IF_FALSE(x.get_dtype() == MJTNUM_DTYPE && x.get_nd() == 1 && x.get_flags() & bn::ndarray::C_CONTIGUOUS);
    _PlotInit();
    SetState(reinterpret_cast<const mjtNum*>(x.get_data()),m_model,m_data);
    m_viewer->SetData(m_data);
    m_viewer->Idle();
}


int _ndarraysize(const bn::ndarray& arr) {
  int prod = 1;
  for (int i=0; i < arr.get_nd(); ++i) {
    prod *= arr.shape(i);
  }
  return prod;
}
template<typename T>
void _copyscalardata(const bp::object& from, T& to) {
  to = bp::extract<T>(from);
}
template <typename T>
void _copyarraydata(const bn::ndarray& from, T* to) {
  FAIL_IF_FALSE(from.get_dtype() == bn::dtype::get_builtin<T>() && from.get_flags() & bn::ndarray::C_CONTIGUOUS);
  memcpy(to, from.get_data(), _ndarraysize(from)*sizeof(T));
}
template<typename T>
void _csdihk(bp::dict d, const char* key, T& to) {
  // copy scalar data if has_key
  if (d.has_key(key)) _copyscalardata(d[key], to);
}
template<typename T>
void _cadihk(bp::dict d, const char* key, T* to) {
  // copy array data if has_key
  if (d.has_key(key)) {
    bn::ndarray arr = bp::extract<bn::ndarray>(d[key]);
    _copyarraydata<T>(arr, to);
  }
}

bp::dict PyMJCWorld2::GetModel() {
    bp::dict out;
    #include "mjcpy2_getmodel_autogen.i"
    return out;
}
void PyMJCWorld2::SetModel(bp::dict d) {
    #include "mjcpy2_setmodel_autogen.i"
}
bp::dict PyMJCWorld2::GetData() {
    bp::dict out;
    #include "mjcpy2_getdata_autogen.i"
    
    return out;
}
void PyMJCWorld2::SetData(bp::dict d) {
    #include "mjcpy2_setdata_autogen.i"
}


}


BOOST_PYTHON_MODULE(mjcpy) {
    bn::initialize();

    bp::class_<PyMJCWorld2,boost::noncopyable>("MJCWorld","docstring here", bp::init<const std::string&>())

        .def("step",&PyMJCWorld2::Step)
        .def("pack_obs",&PyMJCWorld2::PackObs)
        // .def("StepMulti2",&PyMJCWorld::StepMulti2)
        // .def("StepJacobian", &PyMJCWorld::StepJacobian)
        // .def("Plot",&PyMJCWorld::Plot)
        // .def("SetActuatedDims",&PyMJCWorld::SetActuatedDims)
        // .def("ComputeContacts", &PyMJCWorld::ComputeContacts)
        // .def("SetTimestep",&PyMJCWorld::SetTimestep)
        // .def("SetContactType",&PyMJCWorld::SetContactType)
        .def("get_model",&PyMJCWorld2::GetModel)
        .def("set_model",&PyMJCWorld2::SetModel)
        .def("get_data",&PyMJCWorld2::GetData)
        .def("set_data",&PyMJCWorld2::SetData)
        .def("plot",&PyMJCWorld2::Plot)
        .def("idle",&PyMJCWorld2::Idle)
        .def("get_COM_multi",&PyMJCWorld2::GetCOMMulti)
        .def("get_jac_site",&PyMJCWorld2::GetJacSite)
        .def("kinematics",&PyMJCWorld2::Kinematics)
        // .def("SetModel",&PyMJCWorld::SetModel)
        // .def("GetImage",&PyMJCWorld::GetImage)
        .def("set_num_steps",&PyMJCWorld2::SetNumSteps)
        ;


    bp::object main = bp::import("__main__");
    main_namespace = main.attr("__dict__");    
    bp::exec(
        "import numpy as np\n"
        "contact_dtype = np.dtype([('dim','i'), ('geom1','i'), ('geom2','i'),('flc_address','i'),('compliance','f8'),('timeconst','f8'),('dist','f8'),('mindist','f8'),('pos','f8',3),('frame','f8',9),('friction','f8',5)])\n"
        , main_namespace
    );


}
