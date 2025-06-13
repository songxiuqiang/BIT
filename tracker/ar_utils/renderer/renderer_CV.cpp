#include "renderer_CV.h"

namespace ar3dv {

	RenderingEngineCV* RenderingEngineCV::m_instance;

    int cmp(const std::pair<size_t, float> &a, const std::pair<size_t, float> &b)
    {
        return a.second > b.second;
    }

    void RenderingEngineCV::init(const cv::Matx33f &K, int width, int height, float zNear, float zFar,
                                 Model* model) {
        this->K = K;

        vertices = model->getVertices();
        indices = model->getIndices();
        faces = model->getFaces();
        faceNormals = model->getFaceNormals();
        faceCenters = model->getFaceCenters();
        minusd = model->getMinusd();

        fgMask = cv::Mat::zeros(height, width, CV_8UC1);
        depth = cv::Mat::zeros(height, width, CV_32FC1);
    }

    void RenderingEngineCV::render(Model* model) {
        cv::Matx44f pose = model->getPose();

        // get camera intrinsic and extrinsic
        cv::Matx33f R = pose.get_minor<3, 3>(0, 0);
        cv::Vec3f t(pose(0, 3), pose(1, 3), pose(2, 3));
        cv::Matx33f KR = K*R;
        cv::Vec3f Kt = K*t;

        // get visible faces
        std::vector<size_t> visibleFacesId;
        getVisibleFaces(model, R, t, visibleFacesId);

        // render mask;
        cv::Point triPoints[3];
        fgMask = cv::Mat::zeros(fgMask.rows, fgMask.cols, CV_8UC1);
        depth = cv::Mat::zeros(depth.rows, depth.cols, CV_32FC1);
        cv::Mat fgFaceIdMask = cv::Mat::zeros(fgMask.rows, fgMask.rows, CV_32FC1);

        for (size_t i = 0; i < visibleFacesId.size(); ++i)
        {
            size_t vfid = visibleFacesId[i];
            const cv::Vec3f &v0 = vertices[faces[vfid][0]];
            const cv::Vec3f &v1 = vertices[faces[vfid][1]];
            const cv::Vec3f &v2 = vertices[faces[vfid][2]];
            cv::Vec3f projectedv0 = KR*v0 + Kt;
            cv::Vec3f projectedv1 = KR*v1 + Kt;
            cv::Vec3f projectedv2 = KR*v2 + Kt;
            triPoints[0] = cv::Point(projectedv0[0] / projectedv0[2], projectedv0[1] / projectedv0[2]);
            triPoints[1] = cv::Point(projectedv1[0] / projectedv1[2], projectedv1[1] / projectedv1[2]);
            triPoints[2] = cv::Point(projectedv2[0] / projectedv2[2], projectedv2[1] / projectedv2[2]);

            //cv::line(fgFaceIdMask, triPoints[0], triPoints[1], cv::Scalar(vfid + 1));
            //cv::line(fgFaceIdMask, triPoints[1], triPoints[2], cv::Scalar(vfid + 1));
            //cv::line(fgFaceIdMask, triPoints[2], triPoints[0], cv::Scalar(vfid + 1));
            cv::fillConvexPoly(fgFaceIdMask, triPoints, 3, cv::Scalar(vfid + 1));
            cv::fillConvexPoly(fgMask, triPoints, 3, cv::Scalar(255));

            // as compare for why cv::line x 3 cannot fill triangle
            //cv::line(fgMask, triPoints[0], triPoints[1], cv::Scalar(255));
            //cv::line(fgMask, triPoints[1], triPoints[2], cv::Scalar(255));
            //cv::line(fgMask, triPoints[2], triPoints[0], cv::Scalar(255));
        }

        // render depth. init version. Render at full image. Can speed up by ROI.
        for (int r=0; r<depth.rows; ++r) {
            for (int c=0; c<depth.cols; ++c) {
                if (fgMask.ptr<uchar>(r)[c] < 255) {
                    continue;
                }

                cv::Point ctrPt2d(c,r);
                int p3dFaceId = static_cast<int>(fgFaceIdMask.ptr<float>(r)[c]) - 1;
                cv::Vec3f ctrPt3D;
                unproject(ctrPt2d, p3dFaceId, const_cast<cv::Matx33f &>(K), R, t, ctrPt3D);
                cv::Vec3f ctrPt3D_C = KR*ctrPt3D + Kt;
                depth.ptr<float>(r)[c] = ctrPt3D_C[2];
            }
        }
    }

    cv::Mat RenderingEngineCV::getMask() {
        return fgMask;
    }

    cv::Mat RenderingEngineCV::getDepth() {
        return depth;
    }

    void RenderingEngineCV::getVisibleFaces(Model *model, cv::Matx33f &R, cv::Vec3f &t,
                                            std::vector<size_t> &visibleFacesId)
    {
        // visible test
        std::vector<std::pair<size_t, float> > faceIdAndZ;
        cv::Vec3f camCenter = -R.inv() * t;

        for (size_t i = 0; i < faces.size(); ++i)
        {
            cv::Vec3f viewDirection = camCenter - faceCenters[i];
            if (viewDirection.ddot(faceNormals[i])>0) // yes, the face is visible
            {
                cv::Vec3f faceCenterInCam = R*faceCenters[i] + t;
                faceIdAndZ.push_back(std::pair<size_t, float>(i, faceCenterInCam[2]));
            }
        }

        // sort the faces in order to a far to near faces
        std::sort(faceIdAndZ.begin(), faceIdAndZ.end(), cmp);

        // copy face indexes
        size_t visibleFacesNum = faceIdAndZ.size();
        visibleFacesId.resize(visibleFacesNum);
        for (size_t i = 0; i < visibleFacesNum; ++i)
        {
            visibleFacesId[i] = faceIdAndZ[i].first;
        }
    }

    void RenderingEngineCV::unproject(cv::Point &p2d, int faceID, cv::Matx33f &K, cv::Matx33f &R,
                                      cv::Vec3f &t, cv::Vec3f &p3d)
    {
        cv::Vec3f camCenter = -R.inv() * t;
        cv::Matx33f KRInv = (K*R).inv();
        cv::Vec3f KRInvP2d = KRInv*cv::Vec3f(p2d.x, p2d.y, 1);

        // face equation: n.*(X,Y,Z) - d = 0
        const cv::Vec3f &n = faceNormals[faceID];
        float d = minusd[faceID];

        float nc = n.ddot(camCenter);
        float nKRInvP2d = n.ddot(KRInvP2d);
        p3d = ((d - nc) / nKRInvP2d) * KRInvP2d + camCenter;
    }

}