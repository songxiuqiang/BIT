#include "frame_grabber.hh"

namespace tk {
FrameGrabber* FrameGrabber::GetFrameGrabber(const std::string& frames) {
	//struct _stat buf;
	struct stat buf;

	FrameGrabber* grabber = NULL;
	//if (0 != _stat(frames.c_str(), &buf)) {
	if (0 != stat(frames.c_str(), &buf)) {
		grabber = new VideoFrameGrabber(0);
		grabber->type = FrameGrabber::LIVE;
	//} else if(_S_IFDIR & buf.st_mode) {
	} else if(S_ISDIR(buf.st_mode)) {
		grabber = new ImageFrameGrabber(frames);
		grabber->type = FrameGrabber::IMAGES;
	} else {
		grabber = new VideoFrameGrabber(frames);
		grabber->type = FrameGrabber::VIDEO;
	}

	if (-1 == grabber->width) {
		delete grabber;
		return NULL;
	}	else {
		return grabber;
	}
}

} // namespace tk