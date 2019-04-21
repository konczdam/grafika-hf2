//=============================================================================================
// Computer Graphics Sample Program: GPU ray casting
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 450
    precision highp float;

	uniform vec3 wLookAt, wRight, wUp;          // pos of eye

	layout(location = 0) in vec2 cCamWindowVertex;	// Attrib Array 0
	out vec3 p;

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";
// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 450
    precision highp float;

	struct Material {
		vec3 ka, kd, ks;
		float  shininess;
		vec3 F0;
		bool rough, reflective;
	};

	struct Light {
		vec3 direction;
		vec3 Le, La;
	};

	struct Hit {
		float t;
		vec3 position, normal;
		int mat;	// material index
	};

	struct Ray {
		vec3 start, dir;
	};

	struct Sphere {
		vec3 center;
		float radius;
	};

	struct Triangle{
		vec3 p1,p2,p3;
		vec3 n;
	};



	const int nMaxObjects = 500;
	const int nMaxTriangles = 40;

	uniform vec3 wEye; 
	uniform Light light;     
	uniform Material materials[2];  // diffuse, specular, ambient ref
	uniform int nObjects;
	uniform Sphere objects[20];
	uniform int nTriangles;
	uniform Triangle triangles[nMaxTriangles];

	in  vec3 p;					// point on camera window corresponding to the pixel
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	Hit intersect(const Sphere object, const Ray ray) {
		Hit hit;
		hit.t = -1;
		vec3 dist = ray.start - object.center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0;
		float c = dot(dist, dist) - object.radius * object.radius;
		float discr = b * b - 4.0 * a * c;
		if (discr < 0)
		   return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2.0 / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0 / a;
		if (t1 <= 0)
		   return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - object.center) / object.radius;
		return hit;
	}

	Hit intersectWithTriangle(const Triangle triangle, const Ray ray){
		Hit hit;
		hit.mat = 1;
		hit.t = -1.0;
		hit.t = dot((triangle.p1 - ray.start), triangle.n) / dot(ray.dir, triangle.n);
		if(hit.t < 0.0)
			return hit;
		hit.position  = ray.start + ray.dir * hit.t;
		hit.normal = triangle.n;
		if(dot(cross((triangle.p2 - triangle.p1) ,(hit.position - triangle.p1)), triangle.n) > 0.0 && 
		   dot(cross((triangle.p3 - triangle.p2) ,(hit.position - triangle.p2)), triangle.n) > 0.0 &&
		   dot(cross((triangle.p1 - triangle.p3) ,(hit.position - triangle.p3)), triangle.n) > 0.0)
			return hit;
		hit.t = -2.0;
		return hit;
}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1;
		for (int o = 0; o < nObjects; o++) {
			Hit hit = intersect(objects[o], ray); //  hit.t < 0 if no intersection
		    hit.mat = 0;	 
		
			
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  
				bestHit = hit;
		
		}
			
		for(int i = 0; i < nTriangles; i++){
			Hit hit = intersectWithTriangle(triangles[i], ray);
						if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  
								bestHit = hit;
		}			




		if (dot(ray.dir, bestHit.normal) > 0) 
			bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (int o = 0; o < nObjects; o++) 
			if (intersect(objects[o], ray).t > 0)
			   return true; //  hit.t < 0 if no intersection
		 for (int o = 0; o < nObjects; o++) {
			if(intersectWithTriangle(triangles[o], ray).t > 0)
				return true;
			return false;
		}
	}

	vec3 Fresnel(vec3 F0, float cosTheta) { 
		return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
	}

	const float epsilon = 0.0001f;
	const int maxdepth = 15;

	vec3 trace(Ray ray) {
		vec3 weight = vec3(1, 1, 1);
		vec3 outRadiance = vec3(0.0, 0.0, 0.0);
		int n = 0;

		for(int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) {
				//outRadiance += light.La;
				//outRadiance += weight * (3*(14-n))/14.0 * light.La;
				outRadiance += weight *  light.La;
				//outRadiance += weight * light.Le * materials[hit.mat].ks;
				break;
				}
			if (materials[hit.mat].rough) {
				outRadiance += weight * materials[hit.mat].ka * light.La;
				Ray shadowRay;
				shadowRay.start = hit.position + hit.normal * epsilon;
				shadowRay.dir = light.direction;
				float cosTheta = dot(hit.normal, light.direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance += weight * light.Le * materials[hit.mat].kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light.direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0)
					 outRadiance += weight * light.Le * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
				}
				break;
			}

			if (materials[hit.mat].reflective) {
				weight *= Fresnel(materials[hit.mat].F0, dot(-ray.dir, hit.normal));
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
				//outRadiance += light.La * 0.09;
				n++;
			} 
			//else 
		}
				return outRadiance;
	}

	void main() {

		Ray ray;
		ray.start = wEye; 
		ray.dir = normalize(p - wEye);
		fragmentColor = vec4(trace(ray), 1); 
	}



)";

class Material {
protected:
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	bool rough, reflective;
public:
	
	void SetUniform(unsigned int shaderProg, int mat) {
		char buffer[256];
		sprintf(buffer, "materials[%d].ka", mat);
		ka.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].kd", mat);
		kd.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].ks", mat);
		ks.SetUniform(shaderProg, buffer);
		sprintf(buffer, "materials[%d].shininess", mat);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, shininess); else printf("uniform material.shininess cannot be set\n");
		sprintf(buffer, "materials[%d].F0", mat);
		F0.SetUniform(shaderProg, buffer);

		sprintf(buffer, "materials[%d].rough", mat);
		location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1i(location, rough ? 1 : 0); else printf("uniform material.rough cannot be set\n");
		sprintf(buffer, "materials[%d].reflective", mat);
		location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1i(location, reflective ? 1 : 0); else printf("uniform material.reflective cannot be set\n");
	}
};

class RoughMaterial : public Material {
public:
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}
};

class SmoothMaterial : public Material {
public:
	SmoothMaterial(vec3 _F0) {
		F0 = _F0;
		rough = false;
		reflective = true;
	}
};

vec3 calculateF0(vec3 n, vec3 k) {
	vec3 res;
	res.x = (powf(n.x - 1.0f, 2.0f) + powf(k.x, 2.0f)) / (powf(n.x + 1.0f, 2.0f) + powf(k.x, 2.0f));
	res.y = (powf(n.y - 1.0f, 2.0f) + powf(k.y, 2.0f)) / (powf(n.y + 1.0f, 2.0f) + powf(k.y, 2.0f));
	res.z = (powf(n.z - 1.0f, 2.0f) + powf(k.z, 2.0f)) / (powf(n.z + 1.0f, 2.0f) + powf(k.z, 2.0f));
	return res;
}

SmoothMaterial* Gold() {
	vec3 F0 = calculateF0(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f));
	return new SmoothMaterial(F0);
}

struct Sphere {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius) { 
		center = _center; 
		radius = _radius; 
	}

	void SetUniform(unsigned int shaderProg, int o) {
		char buffer[256];
		sprintf(buffer, "objects[%d].center", o);
		center.SetUniform(shaderProg, buffer);
		sprintf(buffer, "objects[%d].radius", o);
		int location = glGetUniformLocation(shaderProg, buffer);
		if (location >= 0) glUniform1f(location, radius); else printf("uniform %s cannot be set\n", buffer);
	}
};

struct Triangle {
	vec3 p1, p2, p3;
	vec3 n;

	Triangle(const vec3& _p1, const vec3& _p2, const vec3& _p3) {
		p1 = _p1; p2 = _p2; p3 = _p3;
		n = normalize(cross((p2 - p1), (p3 - p1)));
		
	}

	void SetUniform(unsigned int shaderProg, int o) {
		char buffer[256];
		sprintf(buffer, "triangles[%d].p1", o);
		p1.SetUniform(shaderProg, buffer);
		sprintf(buffer, "triangles[%d].p2", o);
		p2.SetUniform(shaderProg, buffer);
		sprintf(buffer, "triangles[%d].p3", o);
		p3.SetUniform(shaderProg, buffer);
		sprintf(buffer, "triangles[%d].n", o);
		n.SetUniform(shaderProg, buffer);
	}

};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	vec3 getEye() {
		return eye;
	}
	
	void set(vec3 _eye, vec3 _lookat, vec3 vup, double _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tan(fov / 2);
		up = normalize(cross(w, right)) * f * tan(fov / 2);
	}
	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		set(eye, lookat, up, fov);
	}
	void SetUniform(unsigned int shaderProg) {
		eye.SetUniform(shaderProg, "wEye");
		lookat.SetUniform(shaderProg, "wLookAt");
		right.SetUniform(shaderProg, "wRight");
		up.SetUniform(shaderProg, "wUp");
	}
};

struct Light {
	vec3 direction;
	vec3 Le, La;
	Light(vec3 _direction, vec3 _Le, vec3 _La) {
		direction = normalize(_direction);
		Le = _Le; La = _La;
	}
	void SetUniform(unsigned int shaderProg) {
		La.SetUniform(shaderProg, "light.La");
		Le.SetUniform(shaderProg, "light.Le");
		direction.SetUniform(shaderProg, "light.direction");
	}
};


float rnd() { return (float)rand() / RAND_MAX; }

class Scene {
	std::vector<Sphere*> objects;
	std::vector<Light*> lights;
	Camera camera;
	std::vector<Material*> materials;
	std::vector<Triangle*> triangles;
public:
	Camera getCamera() {
		return camera;
	}
	
	void build() {
		vec3 eye = vec3(0.5, 0.5, 11);
		vec3 vup = vec3(0, 1, 0);
		vec3 lookat = vec3(0.5, 0.5, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		lights.push_back(new Light(vec3(-2, 2, 0), vec3(1, 1, 1), vec3(0.9, 0.9, 0.9)));

		vec3 kd(0.3f, 0.2f, 0.1f);
		vec3 ks(1, 1, 1);
		materials.push_back(new RoughMaterial(kd, ks, 50));
		materials.push_back(Gold());
		//for (int i = 0; i < 250; i++) 
			objects.push_back(new Sphere(vec3(0.5, 0.5, 0),  0.2));
		//materials.push_back(new SmoothMaterial(vec3(0.9, 0.85, 0.8)));
	}
	void SetUniform(unsigned int shaderProg) {
		int location = glGetUniformLocation(shaderProg, "nObjects");
		if (location >= 0) 
			glUniform1i(location, objects.size());
		else 
			printf("uniform nObjects cannot be set\n");

		
		for (int i = 0; i < objects.size(); i++)
			objects[i]->SetUniform(shaderProg, i);

		lights[0]->SetUniform(shaderProg);
		camera.SetUniform(shaderProg);

		for (int mat = 0; mat < materials.size(); mat++)
			materials[mat]->SetUniform(shaderProg, mat);
	}
	
	void Animate(float dt) {
		camera.Animate(dt);
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

class MirrorSystemManager {
	int n = 3;
	const float r = 0.55f;
	std::vector<Triangle*> triangles;

public:
	void build() {
		triangles.clear();
		const int minN = 3;
		const int maxN = 10;
		const float z0 = 0.5f;
		const float z1 = scene.getCamera().getEye().z;
		const vec2 centre = vec2(0.5f, 0.5f);
		const float offset = 2 * M_PI / n;
		for (int i = 0; i < n; i++) {
			const float x0 = cosf(i*offset)*r + centre.x;
			const float x1 = x0;
			const float x2 = cosf((i+1)*offset)*r + centre.x;
			const float y0 = sinf(i*offset)*r + centre.y;
			const float y1 = y0;
			const float y2 = sinf((i + 1)*offset)*r + centre.y;
			triangles.push_back(new Triangle(vec3(x0, y0, z0), vec3(x1, y1, z1), vec3(x2, y2, z1)));
			triangles.push_back(new Triangle(vec3(x2, y2, z0), vec3(x0, y0, z0), vec3(x2, y2, z1)));
		}

	}
		void increaseN() {
			if (n < 20) {
				n++;
				build();
			}
	}

	void SetUniform(unsigned int shaderProg) {
		int location = glGetUniformLocation(shaderProg, "nTriangles");
		if (location >= 0)
			glUniform1i(location, triangles.size());
		else
			printf("uniform nTriangles cannot be set\n");

		for(int i = 0; i < triangles.size(); i++)
			triangles[i]->SetUniform(shaderProg, i);
	}
};
class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
public:
	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad fullScreenTexturedQuad;
MirrorSystemManager mrs;
// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	mrs.build();
	fullScreenTexturedQuad.Create();

	// create program for the GPU
	gpuProgram.Create(vertexSource, fragmentSource, "fragmentColor");
	gpuProgram.Use();
}

bool rotate = false;
// Window has become invalid: Redraw
void onDisplay() {
	static int nFrames = 0;
	nFrames++;
	static long tStart = glutGet(GLUT_ELAPSED_TIME);
	long tEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("%d msec\r", (tEnd - tStart) / nFrames);

	glClearColor(1.0f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.SetUniform(gpuProgram.getId());
	mrs.SetUniform(gpuProgram.getId());
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ') {
		rotate = !rotate;
	}
	if (key == 'a') {
		mrs.increaseN();
	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) {}
void onMouse(int button, int state, int pX, int pY) {}
void onMouseMotion(int pX, int pY) {}

void onIdle() {
	if(rotate)
		scene.Animate(0.01);
	glutPostRedisplay();
}
